# Sanity checks
from numpy import place
import gpt2
from transformers import (
    GPT2ForQuestionAnswering,
    GPT2Tokenizer,
)
import torch
from torch import nn

torch.manual_seed(0)

x = torch.rand(100)

fake_q = gpt2.fake_quantize(x, num_bits=8, per_channel=False)

# FakeQuantize should make a difference
assert (x - fake_q).norm().item() > 0.0

# Lora


def lora():
    lora_conv1d = gpt2.QuantizedConv1d(20, 10, lora=True)
    lora_conv1d.weight.data = torch.zeros_like(lora_conv1d.weight)

    lora_linear = gpt2.QuantizedLinear(10, 20, lora=True)
    x = torch.rand(2, 10)

    lora_conv1d.lora = False
    y1 = lora_conv1d(x)

    lora_conv1d.lora = True
    y2 = lora_conv1d(x)

    assert (y1 - y2).norm().item() == 0

    lora_conv1d.lora_B.data = torch.rand_like(lora_conv1d.lora_B)
    y2 = lora_conv1d(x)
    assert (y1 - y2).norm().item() > 0.0

    o = lora_linear(torch.rand(2, 10))


lora()


def placebo():
    class PlaceboPolicy(gpt2.Policy):
        def get_policy(self, layer_name, layer):
            return gpt2.PolicyDict(
                w_bits=32,
                a_bits=32,
                lora=True,
                rank=10,
                alpha=10,
            )

    orig_model = GPT2ForQuestionAnswering.from_pretrained("./gpt2-squad-finetuned")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    x = tokenizer("hello", "hello", return_tensors="pt")

    y_orig = orig_model(**x, return_dict=True)

    gpt2.apply_qat_to_gpt2(orig_model, policy=PlaceboPolicy())
    y_quant = orig_model(**x, return_dict=True)

    # Default LoRA should not change anything
    assert (y_orig.start_logits - y_quant.start_logits).norm().item() == 0.0


placebo()


def shared_weights():
    orig_linear = nn.Linear(10, 10)
    orig_linear.weight.data = torch.rand_like(orig_linear.weight)

    orig_weight = orig_linear.weight.clone()

    quant_linear = gpt2.QuantizedLinear.from_module(orig_linear, w_bits=8, a_bits=8)

    # Simulate an weight update
    opt = torch.optim.SGD(quant_linear.parameters(), lr=10)
    loss_fn = nn.MSELoss()

    x = torch.rand(2, 10)
    y_hat = torch.rand(2, 10)

    opt.zero_grad()
    y = quant_linear(x)
    loss = loss_fn(y, y_hat)
    loss.backward()
    opt.step()

    # We should have shared weights
    assert (orig_weight - orig_linear.weight).norm().item() > 0.0
    print("Shared diff:", (orig_weight - quant_linear.weight).norm().item())
    assert (orig_linear.weight - quant_linear.weight).norm().item() == 0.0


shared_weights()

# Simulate Cascade Distillation


def cascade_distillation():
    orig_linear = nn.Linear(10, 10)
    orig_linear.weight.data = torch.rand_like(orig_linear.weight)

    orig_weight = orig_linear.weight.clone()

    quant_linear = gpt2.QuantizedLinear.from_module(orig_linear, w_bits=8, a_bits=8)

    opt = torch.optim.SGD(quant_linear.parameters(), lr=10)
    loss_0 = nn.MSELoss()
    loss_1 = nn.MSELoss()

    x = torch.rand(2, 10)
    y_hat = torch.rand(2, 10)

    opt.zero_grad()
    y0 = orig_linear(x)
    y1 = quant_linear(x)
    loss = loss_0(y0, y_hat) + loss_1(y0, y1)
    loss.backward()
    opt.step()

    # We should have shared weights
    assert (orig_weight - orig_linear.weight).norm().item() > 0.0
    print("Cascade diff:", (orig_weight - orig_linear.weight).norm().item())
    assert (orig_linear.weight - quant_linear.weight).norm().item() == 0.0


cascade_distillation()


def gpt2_cascade():
    orig_model = GPT2ForQuestionAnswering.from_pretrained("./gpt2-squad-finetuned")
    # focus on the output head
    head = orig_model.qa_outputs  # is a Linear
    head_orig_weight = head.weight.clone()

    quant_model = gpt2.shallow_clone(orig_model)
    assert quant_model.qa_outputs.weight is head.weight
    gpt2.apply_qat_to_gpt2(quant_model)

    opt = torch.optim.SGD(orig_model.parameters(), lr=10)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    x = tokenizer("hello", "hello", return_tensors="pt")

    start_positions_y_hat = torch.tensor([0])
    end_positions_y_hat = torch.tensor([1])

    y0 = orig_model(
        **x,
        start_positions=start_positions_y_hat,
        end_positions=end_positions_y_hat,
        return_dict=True,
    )

    y1 = quant_model(
        **x,
        start_positions=start_positions_y_hat,
        end_positions=end_positions_y_hat,
        return_dict=True,
    )

    opt.zero_grad()
    total_loss = y0.loss + y1.loss
    print("Total loss:", total_loss.item())
    total_loss.backward()
    opt.step()

    assert (head_orig_weight - head.weight).norm().item() > 0.0
    print("Cascade diff in LLM Head:", (head_orig_weight - head.weight).norm().item())


gpt2_cascade()
