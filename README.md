## Baseline GPT2 Finetuned on SQUADv1

Size of validation set: 10570
Exact: 0.6399243140964995, F1: 0.7349441432312024

## Baseline GPT2 PTQ 4bit and 8bit

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

CrossEntropy for 32-bit + KLDivLoss for distillation losses. Hyper-params applied to each dist loss.

Only train the 32-bit weights. Fix other quantized weights.
