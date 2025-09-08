import torch
import torch.nn as nn
from transformers import (
    Conv1D,
    GPT2ForQuestionAnswering,
)
import copy


class FakeQuantize(torch.autograd.Function):
    """
    Simulates roundings caused by quantization during training.
    MinMax Symmetric as in LLM-QAT
    """

    @staticmethod
    def forward(ctx, x, num_bits=8, per_channel=False) -> torch.Tensor:
        if num_bits >= 32:
            return x  # No quantization needed

        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1

        if per_channel:
            # Calculate scale per output channel
            max_abs_val = x.abs().max(dim=1, keepdim=True).values
        else:
            max_abs_val = x.abs().max()

        scale: torch.Tensor = max_abs_val / qmax
        eps = torch.finfo(scale.dtype).eps
        scale = scale.clamp(min=eps)

        # Quantize and dequantize
        q_x = torch.clamp(torch.round(x / scale), qmin, qmax)
        dq_x = q_x * scale
        return dq_x

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        # Straight-through estimator
        return grad_output, None, None, None

    @classmethod
    def apply(cls, x: torch.Tensor, num_bits=8, per_channel=False) -> torch.Tensor:
        # To improve type checking
        return super().apply(x, num_bits, per_channel)  # type:ignore


class QuantizedLinear(nn.Module):
    """
    Quantized Linear Layer
    """

    @staticmethod
    def from_module(module: nn.Linear, w_bits=8, a_bits=32):
        m = QuantizedLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            w_bits=w_bits,
            a_bits=a_bits,
        )
        m.weight = module.weight
        m.bias = module.bias
        return m

    def __init__(self, in_features, out_features, bias=True, w_bits=8, a_bits=32):
        super().__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        quantized_weight = FakeQuantize.apply(self.weight, self.w_bits, True)

        if self.bias is not None:
            quantized_bias = FakeQuantize.apply(self.bias, self.w_bits, False)
        else:
            quantized_bias = None

        x = FakeQuantize.apply(x, self.a_bits, False)

        return nn.functional.linear(x, quantized_weight, quantized_bias)


class QuantizedConv1d(nn.Module):
    """
    Conv1D is basically a Linear layer with a different shape.
    """

    @staticmethod
    def from_module(module: Conv1D, w_bits=8, a_bits=32):
        m = QuantizedConv1d(nf=module.nf, nx=module.nx, w_bits=w_bits, a_bits=a_bits)
        m.weight = module.weight
        m.bias = module.bias
        return m

    def __init__(self, nf, nx, w_bits=8, a_bits=32):
        super().__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

        self.nf = nf

    def __repr__(self):
        return f"QuantizedConv1d(w_bits={self.w_bits}, a_bits={self.a_bits})"

    def forward(self, x):
        quantized_weight = FakeQuantize.apply(self.weight, self.w_bits, True)
        quantized_bias = FakeQuantize.apply(self.bias, self.w_bits, False)

        x = FakeQuantize.apply(x, self.a_bits, False)

        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(quantized_bias, x.view(-1, x.size(-1)), quantized_weight)
        x = x.view(size_out)

        return x


# Recursively replace Linear layers in GPT-2 with QuantizedLinear
def apply_qat_to_gpt2(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(
                model,
                name,
                QuantizedLinear.from_module(
                    module,
                    w_bits=8,
                    a_bits=32,
                ),
            )
        elif isinstance(module, Conv1D):
            setattr(
                model,
                name,
                QuantizedConv1d.from_module(
                    module,
                    w_bits=8,
                    a_bits=32,
                ),
            )
        else:
            apply_qat_to_gpt2(module)


# Create a shallow cloned version of model. Which all parameters are shared.
def shallow_clone(model: GPT2ForQuestionAnswering) -> GPT2ForQuestionAnswering:
    new_model = copy.deepcopy(model)

    def _clone_shared(m1, m2):
        m1_params = m1.named_parameters(recurse=False)
        m2_params = dict(m2.named_parameters(recurse=False))

        for name, _ in m1_params:
            setattr(m1, name, m2_params[name])

        m1_children = m1.named_children()
        m2_children = dict(m2.named_children())

        for name, child in m1_children:
            _clone_shared(child, m2_children[name])

    _clone_shared(new_model, model)

    return new_model
