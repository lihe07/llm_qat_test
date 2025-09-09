# STEP 1-4

## Baseline GPT2 Finetuned on SQUADv1

Size of validation set: 10570
Exact: 0.6399243140964995, F1: 0.7349441432312024

## Conservative Policy

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

Exact: 0.532, F1: 0.605178903614744

## Conservative LoRA

Before Finetuning

Exact: 0.35960264900662253, F1: 0.471457638051139

With Cascade Distillation. 1000 steps
dist_weight = 1.0
dist_loss = KLDivLoss

Exact: 0.44219489120151373, F1: 0.5520366628615148

## Depth Adaptive Policy

Before Finetuning
Exact: 0.31097445600756857, F1: 0.4168484907718352

With Cascade Distillation. 1000 steps
dist_weight = 1.0
dist_loss = KLDivLoss

Exact: 0.5223273415326396, F1: 0.635513690587542

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
