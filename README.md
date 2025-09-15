# EIC LLM Coding Test

Harvey Li <harvey-l@gatech.edu>

PDF Report: [report.pdf](./report.pdf)

## Requirements

Below is my environment setup for running the code. Similar versions should work.

- Python 3.12.9
- PyTorch 2.8.0 + CUDA 12.8
- Transformers 4.56.1 + Datasets 4.0.0

## Steps to reproduce

1. Fine-tune the full precision model (optional, you can download the [pretrained model](https://www.dropbox.com/scl/fi/7uerce730v8lcsprpe7dt/gpt2-squad-finetuned.zip) directly):

```bash
python train_baseline.py
```

This will save the model to `./gpt2-squad-finetuned`. The folder name do not need to be changed, as it is hard-coded in `qat.py`.

2. Modify and run `qat.py` to reproduce Cyclic Distillation Training (CDT) and Cyclic Precision Training (CPT) results.

Toggle these two lines to switch between CDT and CPT:

```py
my_trainer = Cascade(model, policies_list)
# my_trainer = Cyclic(model, policies_list)
```

Change the `policies_list` to try different quantization policies.

3. Generate HotFlip attack dataset with `attack.py`

You can also download the pre-generated adversarial examples from [here](https://www.dropbox.com/scl/fi/v8kwk4hyysacn65fhcy1l/squad-adversarial.zip).

```bash
python attack.py
```

This will generate `./squad-adversarial/` folder with adversarial examples.

4. Evaluate the model with adversarial examples.

Please first modify `attack_eval.py` to change the policy to the one you want to evaluate.

Then run:

```bash
python ./attack_eval.py
```

## Project Structure

- `train_baseline.py`: Fine-tune GPT-2 on SQuAD dataset with full precision.
- `qat.py`: Client code to run CDT or CPT with different quantization policies.
- `trainer.py`: The parent class for CDT and CPT trainers.
- `cascade.py`: CDT trainer implementation.
- `cyclic.py`: CPT trainer implementation.
- `gpt2.py`: Most of the quantization logic. e.g. FakeQuantize, QuantizedLinear, ...
- `policies.py`: Every quantization policies.
- `attack.py`: Generate adversarial examples using HotFlip attack.
- `attack_eval.py`: Client code to evaluate the model with adversarial examples.

Unused:

- `sanity.py`: Some tests for quantization modules.
- `figures_bars.py` and `figures_pareto.py`: drawing figures in the report.
- `Untitled.ipynb`: some experiments with GPT-2 and quantization.

Other:

- `DATA.md`: Raw output logs. Should contain all the results in the report.
- `report.typ`: The report in Typst format.
