import scipy.stats
from gpt2 import Policy, PolicyDict

import torch.nn as nn

from transformers import Conv1D


class ConservativeStablePolicy(Policy):
    """
    Goals:
      - Minimal risk of quality regression.
      - Use moderate quantization on internal layers; keep final head (qa) unquantized (32).
      - Do NOT apply LoRA. Do full fine-tunning.
    Heuristic bit choices (per-channel weights, per-tensor activations):
      attn_in (QKV fused): w8 / a8 (outlier-prone)
      attn_out:            w7 / a8
      mlp.c_fc (expansion): w6 / a6
      mlp.c_proj:          w7 / a7
      qa_outputs:          w32 / a32 (disable)
    """

    def __init__(self, full_bias=False):
        self.full_bias = full_bias

    def get_policy(self, layer_name, layer):
        t = self.classify(layer_name)
        if t == "qa":
            return PolicyDict(32, 32, 32, False)
        if t == "attn_in":
            return PolicyDict(8, 32 if self.full_bias else 8, 8, False)
        if t == "attn_out":
            return PolicyDict(7, 32 if self.full_bias else 7, 8, False)
        if t == "mlp_fc":
            return PolicyDict(6, 32 if self.full_bias else 6, 6, False)
        if t == "mlp_proj":
            return PolicyDict(7, 32 if self.full_bias else 7, 7, False)
        raise ValueError(f"Unrecognized layer type for {layer_name}")


class ConservativeLoRAPolicy(Policy):
    """
    Same as ConservativeStablePolicy but with LoRA on every layer except head.
    """

    def __init__(self, full_bias=False):
        self.full_bias = full_bias

    def get_policy(self, layer_name, layer):
        t = self.classify(layer_name)
        in_f, out_f = self.get_dims(layer)

        if t == "qa":
            return PolicyDict(32, 32, 32, False)
        if t == "attn_in":
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, "mild", 8)
            return PolicyDict(8, 32 if self.full_bias else 8, 8, True, 8, rank, alpha)
        if t == "attn_out":
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, "mild", 7)
            return PolicyDict(7, 32 if self.full_bias else 7, 8, True, 7, rank, alpha)
        if t == "mlp_fc":
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, "mild", 6)
            return PolicyDict(6, 32 if self.full_bias else 6, 6, True, 6, rank, alpha)
        if t == "mlp_proj":
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, "mild", 7)
            return PolicyDict(7, 32 if self.full_bias else 7, 7, True, 7, rank, alpha)
        raise ValueError(f"Unrecognized layer type for {layer_name}")


class DepthAdaptivePolicy(Policy):
    """
    Depth-aware compression:
      - Early (first 20%) and late (last 20%) layers get +1 bit vs base.
      - Middle layers compressed more.
      - Final qa_outputs: unquantized for numerical stability in output distribution.
    Base bits (mid-layer):
      attn_in: 7/8
      attn_out: 6/7
      mlp_fc: 5/6
      mlp_proj: 6/6
    Adjust +1 to both w_bits and a_bits (cap at 8) for early/late depth bands.
    LoRA only if a layer ends up with w_bits <=4 (which only happens if you manually shrink base).
    """

    def __init__(self, total_layers: int, full_bias=False):
        self.total_layers = total_layers
        self.full_bias = full_bias

    def depth_band(self, depth: int) -> str:
        frac = depth / (self.total_layers - 1)
        if frac <= 0.2:
            return "early"
        if frac >= 0.8:
            return "late"
        return "middle"

    def get_policy(self, layer_name: str, layer):
        t = self.classify(layer_name)
        if t == "qa":
            return PolicyDict(32, 32, 32, False)

        base_map = {
            "attn_in": (7, 8),
            "attn_out": (6, 7),
            "mlp_fc": (5, 6),
            "mlp_proj": (6, 6),
        }
        w_bits, a_bits = base_map.get(t, (7, 7))

        d = self.get_depth(layer_name)
        if d is not None:
            band = self.depth_band(d)
            if band in ("early", "late"):
                w_bits = min(w_bits + 1, 8)
                a_bits = min(a_bits + 1, 8)

        # Decide LoRA: only if extremely low bits (not typical here)
        lora = False
        rank = 1
        alpha = 1
        lora_bits = 32
        if w_bits <= 4 and t in ("attn_out", "mlp_proj"):
            in_f, out_f = self.get_dims(layer)
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, "mild", w_bits)
            lora = True

        return PolicyDict(
            w_bits,
            32 if self.full_bias else w_bits,
            a_bits,
            lora,
            lora_bits,
            rank,
            alpha,
        )


class AggressiveLowBitLoRA(Policy):
    """
    Aggressive compression targeting 4-5 bit weights.
    Strategy:
      - Use LoRA on structurally sensitive projections (attn_out, mlp_proj) and optionally attn_in when bits <=4.
      - Increase activations by +1 bit relative to weights when w_bits <=4 (per-tensor a quant tends to magnify outlier risk).
      - Final qa_outputs kept at 8/8 (can optionally force 32/32 if critical).
    Base (mid-depth):
      attn_in: 5/6
      attn_out: 4/6 (LoRA)
      mlp_fc: 4/5
      mlp_proj: 4/6 (LoRA)
      qa_outputs: 32/32 (no quant)
    Depth modulation:
      Early & late layers: +1 w_bit (cap 8)
      Middle: as base.
    LoRA rank severity classification:
      attn_out, mlp_proj: "aggressive"
      attn_in when w_bits <=4: "moderate"
    """

    def __init__(self, total_layers: int, full_bias=False):
        self.total_layers = total_layers
        self.full_bias = full_bias

    def band(self, depth: int) -> str:
        if self.total_layers <= 1:
            return "middle"
        frac = depth / (self.total_layers - 1)
        if frac <= 0.2:
            return "early"
        if frac >= 0.85:
            return "late"
        return "middle"

    def get_policy(self, layer_name: str, layer):
        t = self.classify(layer_name)
        if t == "qa":
            return PolicyDict(32, 32, 32, False)

        base_map = {
            "attn_in": (5, 8),
            "attn_out": (4, 8),
            "mlp_fc": (4, 8),
            "mlp_proj": (4, 8),
        }
        w_bits, a_bits = base_map.get(t, (5, 8))
        depth = self.get_depth(layer_name)
        if depth is not None:
            b = self.band(depth)
            if b in ("early", "late"):
                w_bits = min(w_bits + 1, 8)

        lora = False
        rank = 1
        alpha = 1
        lora_bits = 32

        in_f, out_f = self.get_dims(layer)
        if t in ("attn_out", "mlp_proj"):
            lora = True
            severity = "aggressive"
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, severity, w_bits)
        elif t == "attn_in" and w_bits <= 4:
            lora = True
            severity = "moderate"
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, severity, w_bits)
        elif t == "mlp_fc" and w_bits <= 4:
            # Optionally apply LoRA if extremely low bits
            lora = True
            severity = "moderate"
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, severity, w_bits)

        return PolicyDict(
            w_bits,
            32 if self.full_bias else w_bits,
            a_bits,
            lora,
            lora_bits,
            rank,
            alpha,
        )


class OutlierAdaptivePolicy(Policy):
    """
    Uses externally provided outlier scores to escalate precision or introduce LoRA.
    outlier_scores: dict[layer_name] = float (e.g., max_channel_norm / median_channel_norm or kurtosis)
    Thresholds:
      high_thr: strong outliers; if w_bits <=6 -> raise bits or add LoRA
      med_thr: mild outliers; +1 bit if below 8
    Base bits (similar to moderate compression):
      attn_in: 6/8
      attn_out: 5/7
      mlp_fc: 4/6
      mlp_proj: 5/7
      qa_outputs: 32/32 (no quant)
    Policies:
      - If high outliers & can't raise bits further (already >=7), add LoRA.
      - Always ensure mlp_proj & attn_out not both at ultra-low bits simultaneously without LoRA.
    """

    def __init__(
        self,
        med_thr: float = 6.0,
        high_thr: float = 10.0,
    ):
        self.med_thr = med_thr
        self.high_thr = high_thr

    def get_policy(self, layer_name: str, layer):
        t = self.classify(layer_name)
        if t == "qa":
            return PolicyDict(32, 32, 32, False)

        weight = layer.weight

        # calculate outlier score
        # for weight, compute the avg per-channel kurtosis
        weight_kurt = 0.0
        for row in weight:
            row_np = row.detach().cpu().numpy()
            kurt = scipy.stats.kurtosis(row_np, fisher=False)
            weight_kurt += kurt
        weight_kurt /= weight.size(0)

        base_map = {
            "attn_in": (6, 8),
            "attn_out": (5, 8),
            "mlp_fc": (4, 8),
            "mlp_proj": (5, 8),
        }
        w_bits, a_bits = base_map.get(t, (6, 7))

        # Escalation rules
        if weight_kurt > self.high_thr:
            print(f"High outlier detected in {layer_name}: kurtosis {weight_kurt:.2f}")
            if w_bits < 7:
                w_bits = min(w_bits + 2, 8)
            else:
                # Add LoRA for refinement
                lora = True
            a_bits = min(a_bits + 1, 8)
        elif weight_kurt > self.med_thr:
            print(
                f"Medium outlier detected in {layer_name}: kurtosis {weight_kurt:.2f}"
            )
            if w_bits < 8:
                w_bits = min(w_bits + 1, 8)
            # Optional mild activation increase if attn layer
            if t in ("attn_in", "attn_out"):
                a_bits = min(a_bits + 1, 8)
        else:
            print(
                f"No significant outliers in {layer_name}: kurtosis {weight_kurt:.2f}"
            )

        # Decide LoRA after adjustments
        lora = False
        rank = 1
        alpha = 1
        lora_bits = 32
        in_f, out_f = self.get_dims(layer)

        # If after escalation still low bits for sensitive layers -> LoRA
        if t in ("attn_out", "mlp_proj") and w_bits <= 5:
            lora = True
            severity = "aggressive" if w_bits <= 4 else "moderate"
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, severity, w_bits)

        # If high_thr triggered but we raised bits above 6, optionally still add a mild LoRA if extreme score
        if (
            weight_kurt > (self.high_thr * 1.5)
            and not lora
            and t in ("attn_out", "attn_in")
        ):
            lora = True
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, "mild", w_bits)

        return PolicyDict(w_bits, w_bits, a_bits, lora, lora_bits, rank, alpha)
