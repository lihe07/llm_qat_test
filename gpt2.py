import abc
from dataclasses import dataclass
import dataclasses
import math
import re
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def from_module(
        module: nn.Linear,
        **kwargs,
    ):
        m = QuantizedLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            **kwargs,
        )
        m.weight = module.weight
        m.bias = module.bias
        return m

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        w_bits=8,
        a_bits=32,
        lora=False,
        rank=1,
        alpha=1,
    ):
        super().__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.lora = lora
        if lora:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

            self.scaling = alpha / rank
            self.weight.requires_grad = False
            self.bias.requires_grad = False

    def forward(self, x):
        quantized_weight = FakeQuantize.apply(self.weight, self.w_bits, True)

        if self.bias is not None:
            quantized_bias = FakeQuantize.apply(self.bias, self.w_bits, False)
        else:
            quantized_bias = None

        x = FakeQuantize.apply(x, self.a_bits, False)
        y = F.linear(x, quantized_weight, quantized_bias)

        if self.lora:
            quantized_lora_A = FakeQuantize.apply(self.lora_A, self.w_bits, True)
            quantized_lora_B = FakeQuantize.apply(self.lora_B, self.w_bits, True)

            lora_update = self.scaling * (quantized_lora_B @ quantized_lora_A)
            y = y + nn.functional.linear(x, lora_update)

        return y


class QuantizedConv1d(nn.Module):
    """
    Conv1D is basically a Linear layer with a different shape.
    """

    @staticmethod
    def from_module(module: Conv1D, **kwargs):
        m = QuantizedConv1d(nf=module.nf, nx=module.nx, **kwargs)
        m.weight = module.weight
        m.bias = module.bias
        return m

    def __init__(self, nf, nx, w_bits=8, a_bits=32, lora=False, rank=1, alpha=1):
        super().__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)
        self.lora = lora

        if lora:
            self.lora_A = nn.Parameter(torch.zeros(rank, nx))
            self.lora_B = nn.Parameter(torch.zeros(nf, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

            self.scaling = alpha / rank
            self.weight.requires_grad = False
            self.bias.requires_grad = False

        self.nf = nf

    def __repr__(self):
        return f"QuantizedConv1d(w_bits={self.w_bits}, a_bits={self.a_bits})"

    def forward(self, x):
        quantized_weight = FakeQuantize.apply(self.weight, self.w_bits, True)
        quantized_bias = FakeQuantize.apply(self.bias, self.w_bits, False)

        x = FakeQuantize.apply(x, self.a_bits, False)

        # Conv1D is transposed Linear
        y = F.linear(x, quantized_weight.t(), quantized_bias)

        if self.lora:
            quantized_lora_A = FakeQuantize.apply(self.lora_A, self.w_bits, True)
            quantized_lora_B = FakeQuantize.apply(self.lora_B, self.w_bits, True)

            lora_update = self.scaling * (quantized_lora_B @ quantized_lora_A)
            y = y + nn.functional.linear(x, lora_update)

        return y


@dataclass
class PolicyDict:
    w_bits: int
    a_bits: int
    lora: bool
    rank: int = 1
    alpha: int = 1


class Policy(abc.ABC):
    """
    All subclasses must implement get_policy().
    Layer names (only valid possibilities):
      - model.transformer.h.x.mlp.c_fc
      - model.transformer.h.x.mlp.c_proj
      - model.transformer.h.x.attn.c_attn
      - model.transformer.h.x.attn.c_proj
      - model.qa_outputs # question answering head
    Depth x: 0..11 (gpt2 small) or 0..23 (gpt2-large).
    """

    LAYER_PATTERNS = {
        "mlp_fc": re.compile(r"\.mlp\.c_fc$"),
        "mlp_proj": re.compile(r"\.mlp\.c_proj$"),
        "attn_in": re.compile(r"\.attn\.c_attn$"),  # fused QKV
        "attn_out": re.compile(r"\.attn\.c_proj$"),
        "qa": re.compile(r"\.qa_outputs$"),
    }

    DEPTH_RE = re.compile(r"\.h\.(\d+)\.")

    def classify(self, layer_name: str) -> str:
        for k, pat in self.LAYER_PATTERNS.items():
            if pat.search(layer_name):
                return k
        raise ValueError(f"Unknown layer name pattern: {layer_name}")

    def get_depth(self, layer_name: str) -> Optional[int]:
        m = self.DEPTH_RE.search(layer_name)
        if m:
            return int(m.group(1))
        return None

    def get_dims(self, layer: Union[nn.Linear, Conv1D]) -> tuple[int, int]:
        """
        Returns (in_features, out_features) for rank heuristics.
        """
        if isinstance(layer, nn.Linear):
            return layer.in_features, layer.out_features
        if isinstance(layer, Conv1D):
            return layer.nx, layer.nf

    def lora_rank_heuristic(
        self, in_f: int, out_f: int, severity: str, w_bits: int
    ) -> tuple[int, int]:
        """
        Heuristic for (rank, alpha)
        severity: one of {"mild", "moderate", "aggressive"} decided by policy usage context.
        Rules:
          - Base rank scales with sqrt(min(in_f, out_f)).
          - More aggressive (lower bits) => higher fraction.
          - Cap rank to a moderate number to avoid high overhead.

        """
        if w_bits == 32:
            return (0, 0)

        dim_ref = min(in_f, out_f)
        base = int((dim_ref**0.5) / 2)  # coarse scaling
        if severity == "mild":
            scale = 0.25
        elif severity == "moderate":
            scale = 0.5
        else:
            scale = 0.75

        # Adjust for bit severity: fewer bits => amplify rank
        if w_bits <= 4:
            scale *= 1.5
        elif w_bits <= 5:
            scale *= 1.2

        rank = max(4, int(base * scale))
        # Soft caps
        # QLoRA shows small ranks (8, 16, 32) is enough even for 4-bit
        # Here we are more conservative than that.
        if dim_ref <= 1024:
            rank = min(rank, 32)
        else:
            rank = min(rank, 48)

        # Common LoRA practice: alpha ~ 2..8 * rank; use 4 * rank unless extremely low bits
        alpha = 4 * rank
        if w_bits <= 4:
            alpha = 6 * rank

        return rank, alpha

    @abc.abstractmethod
    def get_policy(
        self, layer_name: str, layer: Union[nn.Linear, Conv1D]
    ) -> PolicyDict:
        raise NotImplementedError


class DefaultPolicy(Policy):
    def get_policy(self, layer_name, layer):
        if layer_name == "model.qa_outputs":
            return PolicyDict(
                w_bits=32,
                a_bits=32,
                lora=False,
            )
        else:
            return PolicyDict(
                w_bits=8,
                a_bits=8,
                lora=False,
            )


# Recursively replace Linear layers in GPT-2 with QuantizedLinear
def apply_qat_to_gpt2(model: nn.Module, policy: Union[Policy, None] = None):
    if policy is None:
        policy = DefaultPolicy()

    def _apply_qat_to_gpt2(root: str, model: nn.Module, policy: Policy):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                policy_dict = dataclasses.asdict(
                    policy.get_policy(root + "." + name, module)
                )
                setattr(
                    model,
                    name,
                    QuantizedLinear.from_module(
                        module,
                        **policy_dict,
                    ),
                )
            elif isinstance(module, Conv1D):
                policy_dict = dataclasses.asdict(
                    policy.get_policy(root + "." + name, module)
                )
                setattr(
                    model,
                    name,
                    QuantizedConv1d.from_module(
                        module,
                        **policy_dict,
                    ),
                )
            else:
                _apply_qat_to_gpt2(root + "." + name, module, policy)

    _apply_qat_to_gpt2("model", model, policy)


# Create a shallow cloned version of model. Which all parameters are shared.
def shallow_clone(model: nn.Module) -> nn.Module:
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
