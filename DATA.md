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

# All TOGETHER with Cyclic

After QAT:
Policy: PlaceboPolicy
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:30
Exact: 0.6031220435193945, F1: 0.7082424628501578
Policy: OutlierAdaptivePolicy
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:06:58
Exact: 0.5567644276253548, F1: 0.6644210782619213
Policy: AggressiveLowBitLoRA
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:15:18
Exact: 0.5267738883632923, F1: 0.6361663208207623
Policy: DepthAdaptivePolicy
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:47
Exact: 0.47540208136234624, F1: 0.5921434591755828
Policy: ConservativeLoRAPolicy
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:20:09
Exact: 0.4705771050141911, F1: 0.5890252635963865
Policy: UniformLoRAPolicy(w_bits=8, a_bits=8, b_bits=32, lora=True, lora_bits=8)
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:20:53
Exact: 0.5725638599810785, F1: 0.6787266457314
Policy: UniformLoRAPolicy(w_bits=6, a_bits=6, b_bits=32, lora=True, lora_bits=6)
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:20:45
Exact: 0.2683065279091769, F1: 0.37752535950545946
Policy: UniformLoRAPolicy(w_bits=4, a_bits=4, b_bits=32, lora=True, lora_bits=4)
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:20:41
Exact: 0.0012298959318826868, F1: 0.05857159369932814
python ./qat.py  68129.78s user 10430.07s system 1126% cpu 1:56:16.17 total

Selected

Policy: UniformLoRAPolicy(w_bits=8, a_bits=8, b_bits=32, lora=False, lora_bits=8)
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:58
Exact: 0.5815515610217596, F1: 0.6851512702238174
Policy: UniformLoRAPolicy(w_bits=7, a_bits=7, b_bits=32, lora=False, lora_bits=7)
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:56
Exact: 0.5469252601702933, F1: 0.6533543781229495
Policy: UniformLoRAPolicy(w_bits=6, a_bits=6, b_bits=32, lora=False, lora_bits=6)
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:55
Exact: 0.3408703878902554, F1: 0.45991838508843064
Policy: UniformLoRAPolicy(w_bits=5, a_bits=5, b_bits=32, lora=False, lora_bits=5)
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:54
Exact: 0.012109744560075686, F1: 0.06706515166597793
Policy: UniformLoRAPolicy(w_bits=4, a_bits=4, b_bits=32, lora=False, lora_bits=4)
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:43
Exact: 0.0008514664143803217, F1: 0.049804380467493935

# Basic shared params

After QAT:
Policy: PlaceboPolicy
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:31
Exact: 0.6350047303689688, F1: 0.7326418382986938
Policy: OutlierAdaptivePolicy
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:07:10
Exact: 0.522894985808893, F1: 0.6264125778380173
Policy: AggressiveLowBitLoRA
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:14:06
Exact: 0.45288552507095553, F1: 0.5572679560405046
Policy: DepthAdaptivePolicy
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:47
Exact: 0.3712393566698202, F1: 0.4786692946465182
Policy: ConservativeLoRAPolicy
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:21:39
Exact: 0.3977294228949858, F1: 0.5135047898563451
Policy: UniformLoRAPolicy(w_bits=8, a_bits=8, b_bits=32, lora=True, lora_bits=8)
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:21:55
Exact: 0.6009460737937559, F1: 0.6960120041038855
Policy: UniformLoRAPolicy(w_bits=6, a_bits=6, b_bits=32, lora=True, lora_bits=6)
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:21:33
Exact: 0.13074739829706716, F1: 0.20735539194918615
Policy: UniformLoRAPolicy(w_bits=4, a_bits=4, b_bits=32, lora=True, lora_bits=4)
Putting beans on toast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:21:24
Exact: 0.0015137180700094607, F1: 0.05551969774875486
python ./qat.py  66468.42s user 11455.35s system 1086% cpu 1:59:33.49 total

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

# STEP 6

Full

before: Exact: 0.6399243140964995, F1: 0.7349441432312024
after: Exact: 0.427, F1: 0.5015237065018735

Outlier Adaptive:
before: Exact: 0.5684957426679281, F1: 0.6753506433518305
after: Exact: 0.406, F1: 0.4896104874873398

Aggresive:
before: Exact: 0.5448438978240303, F1: 0.6535098448899275
after: Exact: 0.404, F1: 0.4827921288006698

Depth adaptive:
before: Exact: 0.5315988647114475, F1: 0.643391701663484
after: Exact: 0.374, F1: 0.458763062475554

Conservative
before: Exact: 0.5077578051087985, F1: 0.6262254536882265
after: Exact: 0.382, F1: 0.4600417356380782

Uniform 8
before: Exact: 0.5898770104068117, F1: 0.6860145167445376
after: Exact: 0.411, F1: 0.48472784017113146

UniformLoRAPolicy(6, 6, 32, True, 6),
before: Exact: 0.3486281929990539, F1: 0.4718602772535547
after: Exact: 0.293, F1: 0.3684847900604632

UniformLoRAPolicy(4, 4, 32, True, 4),
before: Exact: 0.0008514664143803217, F1: 0.05615857881741222
after: Exact: 0.001, F1: 0.04127139437984762
