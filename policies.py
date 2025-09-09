import abc
import re
from dataclasses import dataclass
from typing import Union, Optional
from gpt2 import Policy, PolicyDict

import torch.nn as nn

from transformers import Conv1D


# ---------------------------
# POLICY 1: ConservativeStablePolicy
# ---------------------------
class ConservativeStablePolicy(Policy):
    """
    Goals:
      - Minimal risk of quality regression.
      - Use moderate quantization on internal layers; keep final head (qa) unquantized (32).
      - Do NOT apply LoRA unless w_bits <=5 on high-impact projections (rare here).
    Heuristic bit choices (per-channel weights, per-tensor activations):
      attn_in (QKV fused): w8 / a8 (outlier-prone)
      attn_out:            w7 / a8
      mlp.c_fc (expansion): w6 / a6
      mlp.c_proj:          w7 / a7
      qa_outputs:          w32 / a32 (disable)
    """

    def get_policy(self, layer_name: str, layer):
        t = self.classify(layer_name)
        if t == "qa":
            return PolicyDict(32, 32, False)
        if t == "attn_in":
            return PolicyDict(8, 8, False)
        if t == "attn_out":
            return PolicyDict(7, 8, False)
        if t == "mlp_fc":
            return PolicyDict(6, 6, False)
        if t == "mlp_proj":
            return PolicyDict(7, 7, False)
        # Fallback (should not normally occur)
        return PolicyDict(8, 8, False)


# ---------------------------
# POLICY 2: DepthAdaptivePolicy
# ---------------------------
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

    def __init__(self, total_layers: int):
        self.total_layers = total_layers

    def depth_band(self, depth: int) -> str:
        if self.total_layers <= 1:
            return "middle"
        frac = depth / (self.total_layers - 1)
        if frac <= 0.2:
            return "early"
        if frac >= 0.8:
            return "late"
        return "middle"

    def get_policy(self, layer_name: str, layer):
        t = self.classify(layer_name)
        if t == "qa":
            return PolicyDict(32, 32, False)

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
        if w_bits <= 4 and t in ("attn_out", "mlp_proj"):
            in_f, out_f = self.get_dims(layer)
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, "mild", w_bits)
            lora = True

        return PolicyDict(w_bits, a_bits, lora, rank, alpha)


# ---------------------------
# POLICY 3: AggressiveLowBitLoRA
# ---------------------------
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
      qa_outputs: 8/8
    Depth modulation:
      Early & late layers: +1 w_bit (cap 8)
      Middle: as base.
    LoRA rank severity classification:
      attn_out, mlp_proj: "aggressive"
      attn_in when w_bits <=4: "moderate"
    """

    def __init__(self, total_layers: int):
        self.total_layers = total_layers

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
            return PolicyDict(8, 8, False)

        base_map = {
            "attn_in": (5, 6),
            "attn_out": (4, 6),
            "mlp_fc": (4, 5),
            "mlp_proj": (4, 6),
        }
        w_bits, a_bits = base_map.get(t, (5, 6))
        depth = self.get_depth(layer_name)
        if depth is not None:
            b = self.band(depth)
            if b in ("early", "late"):
                w_bits = min(w_bits + 1, 8)

        # Activation safety margin: if w_bits <=4 ensure a_bits >= w_bits + 1 (cap 8)
        if w_bits <= 4:
            a_bits = max(a_bits, min(w_bits + 1, 8))

        lora = False
        rank = 1
        alpha = 1

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

        return PolicyDict(w_bits, a_bits, lora, rank, alpha)


# ---------------------------
# POLICY 4: OutlierScoreAdaptivePolicy
# ---------------------------
class OutlierScoreAdaptivePolicy(Policy):
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
        outlier_scores: dict[str, float],
        med_thr: float = 3.0,
        high_thr: float = 6.0,
    ):
        self.scores = outlier_scores
        self.med_thr = med_thr
        self.high_thr = high_thr

    def get_policy(self, layer_name: str, layer):
        t = self.classify(layer_name)
        if t == "qa":
            return PolicyDict(32, 32, False)

        base_map = {
            "attn_in": (6, 8),
            "attn_out": (5, 7),
            "mlp_fc": (4, 6),
            "mlp_proj": (5, 7),
        }
        w_bits, a_bits = base_map.get(t, (6, 7))
        score = self.scores.get(layer_name, 0.0)

        # Escalation rules
        if score > self.high_thr:
            if w_bits < 7:
                w_bits = min(w_bits + 2, 8)
            else:
                # Add LoRA for refinement
                lora = True
            a_bits = min(a_bits + 1, 8)
        elif score > self.med_thr:
            if w_bits < 8:
                w_bits = min(w_bits + 1, 8)
            # Optional mild activation increase if attn layer
            if t in ("attn_in", "attn_out"):
                a_bits = min(a_bits + 1, 8)

        # Decide LoRA after adjustments
        lora = False
        rank = 1
        alpha = 1
        in_f, out_f = self.get_dims(layer)

        # If after escalation still low bits for sensitive layers -> LoRA
        if t in ("attn_out", "mlp_proj") and w_bits <= 5:
            lora = True
            severity = "aggressive" if w_bits <= 4 else "moderate"
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, severity, w_bits)

        # If high_thr triggered but we raised bits above 6, optionally still add a mild LoRA if extreme score
        if score > (self.high_thr * 1.5) and not lora and t in ("attn_out", "attn_in"):
            lora = True
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, "mild", w_bits)

        return PolicyDict(w_bits, a_bits, lora, rank, alpha)


# ---------------------------
# POLICY 5: BitBudgetPolicy
# ---------------------------
class BitBudgetPolicy(Policy):
    """
    Distributes a global approximate weight-bit budget across layers.
    Approach:
      - Each matrix contributes cost ~ (in_dim * out_dim * w_bits).
      - Start from a high default (8 bits) then greedily reduce bits on least-sensitive categories:
          Priority order for REDUCTION (lowest sensitivity first):
            1. mlp_fc
            2. mlp_proj
            3. attn_out
            4. attn_in
            5. qa (never reduced below 8; can stay 32 if budget allows)
      - When a layer goes below 5 bits and is a projection (attn_out/mlp_proj) attach LoRA.
      - Activations: tie to weight bits but keep a_bits >= min(w_bits, 6) for attention.
    NOTE: For simplicity we require a prepass to register layer dimensions via register_layers().
    """

    REDUCTION_ORDER = ["mlp_fc", "mlp_proj", "attn_out", "attn_in"]  # qa excluded

    def __init__(self, target_total_weight_bits: int, keep_qa_full: bool = True):
        self.target_total_weight_bits = target_total_weight_bits
        self.keep_qa_full = keep_qa_full
        self.layer_infos = []  # list of dicts with: name, type, in, out
        self.assigned_w_bits = {}
        self.frozen = False

    def register_layers(
        self, named_modules: list[tuple[str, Union[nn.Linear, Conv1D]]]
    ):
        """
        Collect layer dimension information before computing allocation.
        named_modules: list of (layer_name, module)
        """
        self.layer_infos.clear()
        for name, module in named_modules:
            t = self.classify(name)
            if t is None:
                continue
            if t == "qa" and self.keep_qa_full:
                in_f, out_f = self.get_dims(module)
                self.layer_infos.append(
                    dict(name=name, type=t, in_f=in_f, out_f=out_f, bits=32)
                )
            else:
                in_f, out_f = self.get_dims(module)
                self.layer_infos.append(
                    dict(name=name, type=t, in_f=in_f, out_f=out_f, bits=8)
                )
        self._apply_budget()

    def _total_cost(self) -> int:
        total = 0
        for info in self.layer_infos:
            total += info["in_f"] * info["out_f"] * info["bits"]
        return total

    def _eligible_layers_sorted(self):
        # Return layers (indices) sorted by sensitivity priority for reduction
        priority_map = {t: i for i, t in enumerate(self.REDUCTION_ORDER)}
        candidates = []
        for idx, info in enumerate(self.layer_infos):
            t = info["type"]
            if t == "qa":
                continue
            if info["bits"] > 2:  # minimal floor
                candidates.append((priority_map.get(t, 999), idx))
        candidates.sort()
        return [idx for _, idx in candidates]

    def _apply_budget(self):
        current = self._total_cost()
        # Greedy downward adjustments
        while current > self.target_total_weight_bits:
            changed = False
            for idx in self._eligible_layers_sorted():
                info = self.layer_infos[idx]
                old_bits = info["bits"]
                if old_bits <= 2:
                    continue
                # decrement by 1
                info["bits"] = old_bits - 1
                new_total = self._total_cost()
                if new_total <= current:
                    current = new_total
                    changed = True
                if current <= self.target_total_weight_bits:
                    break
            if not changed:
                # Cannot reduce further
                break
        # Persist assignments
        self.assigned_w_bits = {info["name"]: info["bits"] for info in self.layer_infos}
        self.frozen = True

    def get_policy(self, layer_name: str, layer):
        if not self.frozen:
            raise RuntimeError("Must call register_layers() before querying policy.")
        t = self.classify(layer_name)
        if t is None:
            # Default fallback: 8-bit
            return PolicyDict(8, 8, False)

        w_bits = self.assigned_w_bits.get(layer_name, 8)

        # Activation bits heuristic:
        #   a_bits = min(max(w_bits, 6), 8) for attention
        #   a_bits = min(w_bits, 8) for MLP
        if t in ("attn_in", "attn_out"):
            a_bits = max(w_bits, 6)
            a_bits = min(a_bits, 8)
        elif t == "qa":
            a_bits = 32 if w_bits == 32 else max(w_bits, 8)
        else:
            a_bits = min(w_bits, 8)

        # LoRA injection for very low bits on sensitive projections
        lora = False
        rank = 1
        alpha = 1
        if t in ("attn_out", "mlp_proj") and w_bits <= 5:
            in_f, out_f = self.get_dims(layer)
            severity = "aggressive" if w_bits <= 4 else "moderate"
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, severity, w_bits)
            lora = True
        elif t == "attn_in" and w_bits <= 4:
            in_f, out_f = self.get_dims(layer)
            rank, alpha = self.lora_rank_heuristic(in_f, out_f, "moderate", w_bits)
            lora = True

        return PolicyDict(w_bits, a_bits, lora, rank, alpha)
