import squad
from transformers import GPT2ForQuestionAnswering
import gpt2
import policies

model = GPT2ForQuestionAnswering.from_pretrained(
    "./gpt2-qat-depth-adaptive", device_map="auto"
)

gpt2.apply_qat_to_gpt2(
    model,
    policies.DepthAdaptivePolicy(total_layers=12),
)

squad.eval_attack(model)
