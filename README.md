# STEP 1-4

## Baseline GPT2 Finetuned on SQUADv1

Size of validation set: 10570
Exact: 0.6399243140964995, F1: 0.7349441432312024

## Conservative Policy

Budget 588085312

Before Finetuning

Exact: 0.35960264900662253, F1: 0.471457638051139

Without Cascade Distillation. 1000 steps
dist_weight = 0.0

Exact: 0.5112582781456954, F1: 0.6249025591912254

With Cascade Distillation. 1000 steps
dist_weight = 1.0
dist_loss = KLDivLoss

Exact: 0.5156102175969726, F1: 0.6287690323236491

Custom Estimator T=5

Exact: 0.507, F1: 0.5846562235055388

Full Bias:

Budget 590168128
Exact: 0.532, F1: 0.605178903614744

## Conservative LoRA

Before Finetuning

Exact: 0.35960264900662253, F1: 0.471457638051139

With Cascade Distillation. 1000 steps
dist_weight = 1.0
dist_loss = KLDivLoss

Exact: 0.44219489120151373, F1: 0.5520366628615148

## Depth Adaptive Policy

Budget 545576512

Before Finetuning
Exact: 0.31097445600756857, F1: 0.4168484907718352

With Cascade Distillation. 1000 steps
dist_weight = 1.0
dist_loss = KLDivLoss

Exact: 0.5223273415326396, F1: 0.635513690587542

## Aggresive

Budget 430867264

Before:
Exact: 0.42336802270577106, F1: 0.528154412218754

Exact: 0.5448438978240303, F1: 0.6528876190592499

## Outlier Adaptive

Budget 463372864

Before:
Exact: 0.49725638599810784, F1: 0.6018914627367191

After:

Exact: 0.575212866603595, F1: 0.6800193651930744

# ALL TOGETHER

4h17m18s

Policy: PlaceboPolicy
Bitbudget: 2720612416
Policy: OutlierAdaptivePolicy
Bitbudget: 463372864
Policy: AggressiveLowBitLoRA
Bitbudget: 423838528
Policy: DepthAdaptivePolicy
Bitbudget: 545576512
Policy: ConservativeLoRAPolicy
Bitbudget: 590168128
Policy: UniformLoRAPolicy(w_bits=8, a_bits=8, b_bits=32, lora=True, lora_bits=8)
Bitbudget: 682180672
Policy: UniformLoRAPolicy(w_bits=6, a_bits=6, b_bits=32, lora=True, lora_bits=6)
Bitbudget: 512311360
Policy: UniformLoRAPolicy(w_bits=4, a_bits=4, b_bits=32, lora=True, lora_bits=4)
Bitbudget: 342442048

Before QAT:
Policy: PlaceboPolicy
Exact: 0.6399243140964995, F1: 0.7349441432312024

Policy: OutlierAdaptivePolicy
Exact: 0.49725638599810784, F1: 0.6018914627367191

Policy: AggressiveLowBitLoRA
Exact: 0.42336802270577106, F1: 0.528154412218754

Policy: DepthAdaptivePolicy
Exact: 0.31097445600756857, F1: 0.4168484907718352

Policy: ConservativeLoRAPolicy
Exact: 0.36669820245979184, F1: 0.47965911077171414

Policy: UniformLoRAPolicy(w_bits=8, a_bits=8, b_bits=32, lora=True, lora_bits=8)
Exact: 0.5898770104068117, F1: 0.6860145167445376

Policy: UniformLoRAPolicy(w_bits=6, a_bits=6, b_bits=32, lora=True, lora_bits=6)
Exact: 0.10085146641438032, F1: 0.17114364629037493

Policy: UniformLoRAPolicy(w_bits=4, a_bits=4, b_bits=32, lora=True, lora_bits=4)
Exact: 0.000946073793755913, F1: 0.0578531377676724

After QAT:
Policy: PlaceboPolicy
Exact: 0.5943235572374646, F1: 0.7002484148756903

Policy: OutlierAdaptivePolicy
Exact: 0.5684957426679281, F1: 0.6753506433518305

Policy: AggressiveLowBitLoRA
Exact: 0.5448438978240303, F1: 0.6535098448899275

Policy: DepthAdaptivePolicy
Exact: 0.5315988647114475, F1: 0.643391701663484

Policy: ConservativeLoRAPolicy
Exact: 0.5077578051087985, F1: 0.6262254536882265

Policy: UniformLoRAPolicy(w_bits=8, a_bits=8, b_bits=32, lora=True, lora_bits=8)
Exact: 0.5808893093661306, F1: 0.6867220171120404

Policy: UniformLoRAPolicy(w_bits=6, a_bits=6, b_bits=32, lora=True, lora_bits=6)
Exact: 0.3486281929990539, F1: 0.4718602772535547

Policy: UniformLoRAPolicy(w_bits=4, a_bits=4, b_bits=32, lora=True, lora_bits=4)
Exact: 0.0008514664143803217, F1: 0.05615857881741222

## LLM-QAT + GPT2

LLM-QAT summary:

- MinMax Symmetric Quantization > Clipping

- Distillation: not implemented. just do downstream fine-tune

## InstantNet - Switchable Precision

One unified weights: 32-bit

On the run, weights quantized to 4-bit, 8-bit, 16-bit...

How this weights were trained? Cascade Distillation

Loss = 32-bit loss (normal) + distillation loss 16-bit norm(m32(x) - m16(x)) + distillation loss 8-bit norm(m16(x) - m8(x)) + distillation loss 4-bit norm(m8(x) - m4(x))

Or more precisely:

CrossEntropy for 32-bit + MSE for distillation losses. Hyper-params applied to each dist loss.

Actual impl uses MSE loss. Here I used KLDivLoss.

Only train the 32-bit weights. Fix other quantized weights.

# STEP 5
