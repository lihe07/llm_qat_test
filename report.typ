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

= Performance Under Different Configurationsa

- [Step 4] What is the task accuracy achieved after applying various quantization bit-width configurations to the SQuAD dataset?

- [Step 4] How did you determine the optimal quantization bit-width configurations? Have you gleaned any insights from your observations that could guide future work to further enhance performance?

- [Step 4] A motivation behind switchable quantization is to support diverse layer-wise quantization configurations simultaneously, accommodating different resource allocation needs. Could you suggest additional training objectives that could more effectively facilitate the mechanism for switching quantization bit-widths?

