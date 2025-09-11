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

= Introduction

#lorem(100)


= Cascade Training

- [Step 4] What is the task accuracy achieved after applying various quantization bit-width configurations to the SQuAD dataset?

Policies

Conservative
Conservative Full-Bias
Conservative LoRA

Depth Adaptive

- [Step 4] How did you determine the optimal quantization bit-width configurations? Have you gleaned any insights from your observations that could guide future work to further enhance performance?

I compared the Exact and F1 Scores.

The best policy is the Outlier Adaptive, with a bit budget of only 463372864, 

But it's a trade-off problem, So we need to take the Pareto front into consideration.

Insert a figure of Pareto front here.

Some insights: despite requiring less bits, Outlier Adaptive effectively outperforms the believed Conservative Stable policy.
So it's not necessarily true that more bits lead to better performance.



- [Step 4] A motivation behind switchable quantization is to support diverse layer-wise quantization configurations simultaneously, accommodating different resource allocation needs. Could you suggest additional training objectives that could more effectively facilitate the mechanism for switching quantization bit-widths?

I could imaging introducing Parametric Noise Injection 

= Cyclic Training




= Impact on Robustness


