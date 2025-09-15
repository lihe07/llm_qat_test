import abc
from dataclasses import dataclass
import dataclasses
import math
import re
from typing import Dict, Optional, Union
import torch
from torch.autograd.function import once_differentiable
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Conv1D,
    GPT2ForQuestionAnswering,
)
import copy


T = 40.0


class FakeQuantize(torch.autograd.Function):
    """
    Simulates roundings caused by quantization during training.
    MinMax Symmetric as in LLM-QAT
    """

    @staticmethod
    def forward(x, num_bits=8, per_channel=False):
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
        input_scaled = x / scale
        q_x = torch.clamp(torch.round(input_scaled), qmin, qmax)
        delta = input_scaled - torch.floor(input_scaled) - 0.5

        dq_x = q_x * scale
        return (dq_x, scale, delta)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, scale, delta = output

        ctx.mark_non_differentiable(scale, delta)
        ctx.save_for_backward(scale, delta)

    @staticmethod
    def backward(ctx, grad_output, _0, _1):  # type: ignore
        # Straight-through estimator
        return grad_output, None, None, None
        # Sigmoid STE
        if hasattr(ctx, "not_quantized") and ctx.not_quantized:
            return grad_output, None, None
        scale, delta = ctx.saved_tensors

        grad_mat = torch.exp(T * delta) * T / (1 + torch.exp(T * delta)) ** 2

        grad_mat += 1 / (math.exp(T) + 1)

        # clamp with epsilon to avoid zero gradient
        thresh = torch.finfo(grad_mat.dtype).eps
        grad_mat = torch.clamp(grad_mat, min=thresh)

        return grad_output * grad_mat, None, None

    @classmethod
    def apply(cls, x: torch.Tensor, num_bits=8, per_channel=False) -> torch.Tensor:
        # To improve type checking
        return super().apply(x, num_bits, per_channel)  # type:ignore


def fake_quantize(x: torch.Tensor, num_bits=8, per_channel=False) -> torch.Tensor:
    if num_bits >= 32:
        return x
    return FakeQuantize.apply(x, num_bits, per_channel)[0]


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
        b_bits=8,
        a_bits=32,
        lora=False,
        rank=1,
        alpha=1,
        lora_bits=8,
    ):
        super().__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.b_bits = b_bits
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.lora = lora
        self.lora_bits = lora_bits
        self.rank = rank
        self.alpha = alpha
        if lora:
            self.init_lora_params()

    def init_lora_params(self):
        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = self.alpha / self.rank

    def forward(self, x):
        x = fake_quantize(x, self.a_bits, False)

        if self.lora:
            # use weight/bias.data to avoid modifying original weights
            quantized_weight = fake_quantize(self.weight.data, self.w_bits, True)
            if self.bias is not None:
                quantized_bias = fake_quantize(self.bias.data, self.b_bits, False)
            else:
                quantized_bias = None

            quantized_lora_A = fake_quantize(self.lora_A, self.lora_bits, True)
            quantized_lora_B = fake_quantize(self.lora_B, self.lora_bits, True)

            y = F.linear(x, quantized_weight, quantized_bias)
            lora_update = self.scaling * (quantized_lora_B @ quantized_lora_A)
            y = y + nn.functional.linear(x, lora_update)
        else:
            quantized_weight = fake_quantize(self.weight, self.w_bits, True)

            if self.bias is not None:
                quantized_bias = fake_quantize(self.bias, self.b_bits, False)
            else:
                quantized_bias = None

            y = F.linear(x, quantized_weight, quantized_bias)

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

    def __init__(
        self,
        nf,
        nx,
        w_bits=8,
        b_bits=8,
        a_bits=32,
        lora=False,
        rank=1,
        alpha=1,
        lora_bits=8,
    ):
        super().__init__()
        self.w_bits = w_bits
        self.b_bits = b_bits
        self.a_bits = a_bits
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

        self.nx = nx
        self.nf = nf

        self.lora = lora
        self.rank = rank
        self.lora_bits = lora_bits
        self.alpha = alpha

        if lora:
            self.init_lora_params()

    def init_lora_params(self):
        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.nx))
        self.lora_B = nn.Parameter(torch.zeros(self.nf, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scaling = self.alpha / self.rank

    def __repr__(self):
        return f"QuantizedConv1d(w_bits={self.w_bits}, a_bits={self.a_bits})"

    def forward(self, x):
        x = fake_quantize(x, self.a_bits, False)

        if self.lora:
            quantized_weight = fake_quantize(self.weight.data, self.w_bits, True)
            quantized_bias = fake_quantize(self.bias.data, self.b_bits, False)

            y = F.linear(x, quantized_weight.t(), quantized_bias)

            quantized_lora_A = fake_quantize(self.lora_A, self.lora_bits, True)
            quantized_lora_B = fake_quantize(self.lora_B, self.lora_bits, True)

            lora_update = self.scaling * (quantized_lora_B @ quantized_lora_A)
            lora_update = lora_update.to(x.device)
            y = y + nn.functional.linear(x, lora_update)

        else:
            quantized_weight = fake_quantize(self.weight, self.w_bits, True)
            quantized_bias = fake_quantize(self.bias, self.b_bits, False)

            # Conv1D is transposed Linear
            y = F.linear(x, quantized_weight.t(), quantized_bias)

        return y


@dataclass
class PolicyDict:
    w_bits: int
    b_bits: int
    a_bits: int
    lora: bool
    lora_bits: int = 8
    rank: int = 1
    alpha: int = 1

    def __repr__(self) -> str:
        if self.lora:
            return f"Policy(w_bits={self.w_bits}, b_bits={self.b_bits}, a_bits={self.a_bits}, lora={self.lora}, rank={self.rank}, alpha={self.alpha}, lora_bits={self.lora_bits})"
        else:
            return f"Policy(w_bits={self.w_bits}, b_bits={self.b_bits}, a_bits={self.a_bits}, lora={self.lora})"


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
        """
        Returns one of {"mlp_fc", "mlp_proj", "attn_in", "attn_out", "qa"} if recognizes, else raises.
        """
        for k, pat in self.LAYER_PATTERNS.items():
            if pat.search(layer_name):
                return k
        raise ValueError(f"Unknown layer name pattern: {layer_name}")

    def get_depth(self, layer_name: str) -> Optional[int]:
        """
        Returns depth index if recognizes, else None.
        """
        m = self.DEPTH_RE.search(layer_name)
        if m:
            return int(m.group(1))
        return None

    def get_dims(
        self, layer: Union[nn.Linear, Conv1D, QuantizedConv1d, QuantizedLinear]
    ) -> tuple[int, int]:
        """
        Returns (in_features, out_features) for rank heuristics.
        """
        if isinstance(layer, nn.Linear) or isinstance(layer, QuantizedLinear):
            return layer.in_features, layer.out_features
        if isinstance(layer, Conv1D) or isinstance(layer, QuantizedConv1d):
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
        self,
        layer_name: str,
        layer: Union[nn.Linear, Conv1D, QuantizedLinear, QuantizedConv1d],
    ) -> PolicyDict:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


class PlaceboPolicy(Policy):
    def get_policy(self, layer_name, layer):
        return PolicyDict(
            w_bits=32,
            b_bits=32,
            a_bits=32,
            lora=False,
        )


def get_policy_dict(model: nn.Module, policy: Policy) -> Dict[str, PolicyDict]:
    """
    Pre-compute the policy for each layer in the model. Returns a dict mapping layer names to PolicyDict.
    """
    d = {}

    def _get_policy_dict(root: str, model: nn.Module, policy: Policy):
        for name, module in model.named_children():
            if (
                isinstance(module, nn.Linear)
                or isinstance(module, Conv1D)
                or isinstance(module, QuantizedLinear)
                or isinstance(module, QuantizedConv1d)
            ):
                d[root + "." + name] = policy.get_policy(root + "." + name, module)

            else:
                _get_policy_dict(root + "." + name, module, policy)

    _get_policy_dict("model", model, policy)

    return d


def apply_qat_to_gpt2(
    model: nn.Module, policy: Union[Policy, Dict[str, PolicyDict]] = PlaceboPolicy()
):
    """
    Recursively applies QAT modules to all Linear and Conv1D layers according to the given policy.
    If the layer is already a QuantizedLinear/Conv1d, it modifies its attributes according to the policy.
    """

    def _apply_qat_to_gpt2(root: str, model: nn.Module, policy: Union[Policy, Dict]):
        for name, module in model.named_children():
            key = root + "." + name
            if isinstance(module, nn.Linear):
                policy_dict = dataclasses.asdict(
                    policy.get_policy(key, module)
                    if isinstance(policy, Policy)
                    else policy[key]
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
                    policy.get_policy(key, module)
                    if isinstance(policy, Policy)
                    else policy[key]
                )
                setattr(
                    model,
                    name,
                    QuantizedConv1d.from_module(
                        module,
                        **policy_dict,
                    ),
                )
            elif isinstance(module, QuantizedLinear) or isinstance(
                module, QuantizedConv1d
            ):
                policy_dict = dataclasses.asdict(
                    policy.get_policy(key, module)
                    if isinstance(policy, Policy)
                    else policy[key]
                )
                # Modify existing QuantizedLinear/Conv1d according to policy
                need_lora_init = False
                if not module.lora and policy_dict["lora"]:
                    need_lora_init = True
                for k, v in policy_dict.items():
                    if hasattr(module, k):
                        setattr(module, k, v)
                if need_lora_init:
                    module.init_lora_params()

            else:
                _apply_qat_to_gpt2(root + "." + name, module, policy)

    _apply_qat_to_gpt2("model", model, policy)


def shallow_clone(model: nn.Module) -> nn.Module:
    """
    Creates a shallow clone of the model. All parameters are shared.
    """
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


def calculate_bit_budget(model: nn.Module) -> int:
    """
    Calculates the total bit budget of the model. Counts LoRA parameters only if they cannot be merged.
    """
    total_bits = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinear) or isinstance(module, QuantizedConv1d):
            w_bits = module.w_bits
            b_bits = module.b_bits if module.bias is not None else 0
            lora_bits = module.lora_bits if module.lora else 0
            weight_params = module.weight.numel()
            bias_params = module.bias.numel() if module.bias is not None else 0

            if module.lora and lora_bits != w_bits:
                # Only count LoRA params if they cannot be merged
                lora_A_params = module.lora_A.numel()
                lora_B_params = module.lora_B.numel()
                total_bits += lora_A_params * lora_bits + lora_B_params * lora_bits

            total_bits += weight_params * w_bits + bias_params * b_bits
        elif isinstance(module, nn.Linear):
            # Assume full precision for non-quantized layers
            weight_params = module.weight.numel()
            bias_params = module.bias.numel() if module.bias is not None else 0
            total_bits += weight_params * 32 + bias_params * 32
    return total_bits
