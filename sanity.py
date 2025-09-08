# Sanity checks
import gpt2
from transformers import (
    GPT2ForQuestionAnswering,
    GPT2Tokenizer,
)
import torch
from torch import nn

torch.manual_seed(0)

x = torch.rand(100)

fake_q = gpt2.FakeQuantize.apply(x, num_bits=8, per_channel=False)

# FakeQuantize should make a difference
assert (x - fake_q).norm().item() > 0.0


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
