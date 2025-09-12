#import "@preview/elsearticle:1.0.0": *


#show: elsearticle.with(
  title: "Report on Efficient LLMs via Switchable and Dynamic Quantization",
  authors: (
    (
      name: "Harvey Li",
      affiliation: "Georgia Institute of Technology, Atlanta, USA",
      corr: "harvey-l@gatech.edu",
    ),
  ),
  format: "1p",
  
  // line-numbering: true,
)

= Implementation Details

== GPT2 for SQuAD

Before quantization, I added a question answering head to the GPT2 model, following the implementation in Huggingface's transformers library @transformers @gpt2. The question answering head consists of one linear layer that maps the hidden states of the GPT2 model to the start and end positions of the answer span in the input text.


The modified model is then fine-tuned on the SQuAD v1 dataset @squad for 1 epoch, using the AdamW optimizer with a learning rate of $2 times 10^(-5)$ and a batch size of 4. The fine-tuning process achieves an Exact Match score of 64.0 and an F1 score of 73.5 on the validation set.

All following quantization and training steps are based on this fine-tuned model. If you are having trouble reproducing the results, please use the provided pre-trained model at #link("https://www.dropbox.com/scl/fi/7uerce730v8lcsprpe7dt/gpt2-squad-finetuned.zip").

== LLM-QAT

Every aspect of the original LLM-QAT paper is implemented @llm-qat, except the Data-free Distillation, which was replaced by the Cascade Distillation in InstantNet according to the instructions @instantnet.

LLM-QAT explores different quantization methods, including asymmetric, symmetric, MinMax and outlier clipping methods, and concluded with the choice of MinMax Symmetric quantization. Therefore, for simplicity, I only implemented the MinMax Symmetric quantization method, formulated as follows:

#let vv(x) = $upright(bold(#x))$

$
vv(X)_q = alpha round( vv(X) / alpha ) 
#h(1cm)
alpha = max( |vv(X)| ) / (2^(b-1) - 1)
$

Where $b$ is the bit-width, $X$ is the tensor to be quantized, and $X_q$ is the quantized tensor.

I also tried to play with different gradient estimators, but the Straight-Through Estimator (STE) remains the best choice.

== Cascade Distillation Training (CDT)

Cascade Distillation Training (CDT) is implemented following the instructions in InstantNet @instantnet. CDT allows all bit-width configurations to be trained simultaneously, by balancing the hard loss and distillation loss from higher bit-width configurations. The original loss function is formulated as follows:

$
L = L_"hard" ( Q_i (x), hat(y)) + beta sum_(j=i+1)^(N-1) L_"distill" (Q_i (x), Q_j (x))
$

Note that the teacher model $Q_j$ is not updated when training the student model $Q_i$. The implementation drops the gradient of the teacher model to avoid unnecessary computation.

The loss function choice in the original paper is Cross Entropy Loss for the hard loss and Mean Squared Error for the distillation loss @instantnet. However, I replaced the distillation loss with Kullback-Leibler Divergence (KLDivLoss) to better align with the original knowledge distillation paper @hinton. The hard loss remains as Cross Entropy Loss.

$beta = 1$, following the distill_weight in the original implementation at #link("https://github.com/GATECH-EIC/InstantNet/blob/main/config_train.py")

In my implementation, I maintained one instance of full precision model, and dynamically switched the quantization configurations during training. This is more memory efficient than maintaining multiple instances of the model, and allows for parameter sharing among different configurations.

== Cyclic Precision Training (CPT)

#show table.cell.where(y: 0): strong
#set table(
  stroke: (x, y) => if y == 0 {
    (bottom: 0.7pt + black)
  },
  align: (x, y) => (
    if x > 0 { center }
    else { left }
  )
)




= Quantization Policies


#figure(
  caption: [Configurations of Different Quantization Policies],
  placement: auto,
  table(
    columns: 5,

    // Header Row
    [Policy], [weight], [activation], [bias], [LoRA weights],

    // Data Rows
  [Outlier Adaptive], [Adaptive (4-8 bits)], [8 bits], [Adaptive (4-8 bits)], [32 bits \ (if active)],
  [Aggressive LoRA], [Adaptive (4-6 bits)], [8 bits], [Adaptive (4-6 bits)], [32 bits],
  [Depth Adaptive], [Adaptive (5-8 bits)], [Adaptive (6-8 bits)], [Adaptive (5-8 bits)], [],
  [Conservative LoRA], [6-8 bits], [6-8 bits], [32 bits], [6-8 bits],
  [Uniform LoRA 8], [8 bits], [8 bits], [32 bits], [8 bits], 
  [Uniform LoRA 6], [6 bits], [6 bits], [32 bits], [6 bits], 
  [Uniform LoRA 4], [4 bits], [4 bits], [32 bits], [4 bits],
  )
) <tab:configurations>



7 different quantization configurations are explored. The configurations are summarized in Table @tab:configurations. The following sections describe the heuristics of each dynamic configuration.

== Outlier Adaptive

This policy uses kurtosis to measure the outlier level of each layer, and assigns higher bit-widths to layers with higher outlier levels. The intuition is that outliers in weights can significantly harm the precision of quantization by increasing the scaling factor $alpha$, thus using higher bit-widths for layers with more outliers can help mitigate this issue. @llm-qat

Layers are first assigned with a base bit-width according to their position in the network:

- Question Answering Head: Not quantized (32 bits)
- Attention Input Projections (QKV fused): 6 bits weights, 8 bits activations
- Attention Output Projections: 5 bits weights, 8 bits activations
- MLP Input Projections: 4 bits weights, 8 bits activations
- MLP Output Projections: 5 bits weights, 8 bits activations

Biases are quantized to the same bit-width as their corresponding weights. Then, layers are categorized into three levels of outlier-ness based on their kurtosis values:

1. Low outlier level (kurtosis < 6): No change
2. Medium outlier level (6 $<=$ kurtosis < 10): Increase weight bit-width by 1
3. High outlier level (kurtosis $>=$ 10): Increase weight bit-width by 2

32-bit LoRA is applied as refinement for sensitive layers when their bit-width is less than 6 bits, and for attention layers if they have excessive outliers (kurtosis $>=$ 15).

== Depth Adaptive

In depth adaptive policy, early ($<= 20%$) and late ($>= 80%$) layers have $+1$ bit-width for both weights and activations, while middle layers have the base bit-width. The intuition is that early layers are responsible for extracting low-level features, while late layers are responsible for high-level features, both of which are sensitive for model performance. Middle layers, on the other hand, can afford to be quantized more aggressively.

Question Answering Head is not quantized (32 bits). Biases are quantized to the same bit-width as their corresponding weights. 
LoRA is not applied in this policy.

== Aggressive LoRA

Similar to Outlier Adaptive, but with lower base bit-widths and lower LoRA ranks.

== Conservative LoRA

Conservative LoRA policy manually assigns higher bit-widths (7 or 8 bits) to sensitive layers, and 6 bits to the rest. All layers except the head have quantized LoRA with mild ranks.


= Results with CDT

This section answers the questions for Step 1-4.



- [Step 4] What is the task accuracy achieved after applying various quantization bit-width configurations to the SQuAD dataset?

#let mb(x) = calc.round(x / 8 / 1024 / 1024, digits: 2)


#figure(
  caption: [Performance of Different Quantization Policies with CDT],
  placement: auto,
  table(
    columns: (2.5fr, 2fr, 1fr, 1fr),

    // Header Row
    [QAT Policy], [Size (MB)], [Exact], [F1],

    // Data Rows
    [#emph("Full Precision (after QAT)")], [#emph([#mb(2720612416)])], [#emph([0.5943])], [#emph([0.7002])],
    [Outlier Adaptive], [#mb(463372864)],            [0.5685], [0.6754],
    [Aggressive LoRA], [#mb(423838528)],             [0.5448], [0.6535],
    [Depth Adaptive], [#mb(545576512)],              [0.5316], [0.6434],
    [Conservative LoRA], [#mb(590168128)],           [0.5078], [0.6262],
    [Uniform LoRA (8-bit)], [#mb(682180672)],        [#strong([0.5809])], [#strong([0.6867])],
    [Uniform LoRA (6-bit)], [#mb(512311360)],        [0.3486], [0.4719],
    [Uniform LoRA (4-bit)], [#strong([#mb(342442048)])],        [0.0009], [0.0562],
  )
) <tab:results>

The detailed results are summarized in Table @tab:results. The size is calculated based on the bit-width configurations and the shape of parameters in each layer. The Exact Match and F1 scores are evaluated on the SQuAD v1 validation set.

- [Step 4] How did you determine the optimal quantization bit-width configurations? Have you gleaned any insights from your observations that could guide future work to further enhance performance?

#figure(
  placement: auto,
  image("./figures/pareto_fronts.png"),
  caption: [Pareto Fronts for Policies with CDT]
) <fig:pareto>

  In fact, the optimal quantization bit-width configuration is determined by the trade-off between model size and performance. To visualize this trade-off, Table @tab:results is replotted in a Pareto Front graph, as shown in Figure @fig:pareto.

  One can see that three policies stand out in the Pareto Front: Outlier Adaptive, Uniform LoRA (8-bit), and Aggressive LoRA. These three policies offer the best trade-off between size and performance, and are therefore considered optimal.
 
  An important observation is that Outlier Adaptive policy, which uses the least amount of bits, outperforms the believed best Conservative LoRA policy by a large margin (0.57 v.s. 0.51 exact scores). This suggests that more bits do not necessarily lead to better performance, and that intelligent allocation of bits can yield better results.
  
#figure(
  caption: [Comparison of Different Quantization Policies without CDT],
  placement: auto,
  table(
    columns: 5,

    // Header Row
    [QAT Policy], [Exact], [F1], [Exact \ (with CDT)], [F1 \ (with CDT)],

    // Data Rows
    [Outlier Adaptive],  [#strong([0.5752])], [#strong([0.6800])], [0.5685], [0.6754],
    [Aggressive LoRA],   [0.5448], [0.6529], [#strong([0.5448])], [#strong([0.6535])],
    [Depth Adaptive],    [0.5223], [0.6355], [#strong([0.5316])], [#strong([0.6434])],
    [Conservative LoRA], [0.4422], [0.5520], [#strong([0.5078])], [#strong([0.6262])],
  )
) <tab:results_no_cdt>

  I also experimented with training the same quantization policies without Cascade Distillation Training (CDT) and using only the hard loss. The training hyperparameters remain the same, and the results are summarized in Table @tab:results_no_cdt. One can see that CDT consistently improves the performance of all policies, with the most significant improvement seen in the Conservative LoRA policy (0.51 v.s. 0.44 exact scores). This suggests that CDT is an effective training strategy for switchable quantization, as it allows the model to learn from higher bit-width configurations and transfer knowledge to lower bit-width configurations @instantnet.

- [Step 4] A motivation behind switchable quantization is to support diverse layer-wise quantization configurations simultaneously, accommodating different resource allocation needs. Could you suggest additional training objectives that could more effectively facilitate the mechanism for switching quantization bit-widths?

  In addition to the hard loss and distillation loss used in CDT, one could also consider adding a regularization term that encourages the model to learn similar representations across different bit-width configurations. This could be achieved by minimizing the distance between the hidden states of the model under different quantization configurations, such as using Mean Squared Error or Cosine Similarity as the distance metric. 

  I would also suggest introducing Parametric Noise Injection (PNI) @pni during training, which adds learnable noise to the weights and activations. Quantization noise resembles random noise @double, and PNI can help the model learn to be robust against such noise.


= Results with CPT




= Impact on Robustness


= Conclusion and Future Work

#bibliography("./refs.bib")


