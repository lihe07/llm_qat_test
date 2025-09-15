import squad
from transformers import GPT2ForQuestionAnswering
import gpt2
import safetensors.torch
import policies

model = GPT2ForQuestionAnswering.from_pretrained(
    "./gpt2-squad-finetuned/", device_map="auto"
)


# policies.OutlierAdaptivePolicy(),
# policies.AggressiveLowBitLoRA(total_layers=12),
# policies.DepthAdaptivePolicy(total_layers=12),
# policies.ConservativeLoRAPolicy(full_bias=True),
# policies.UniformLoRAPolicy(8, 8, 32, True, 8),
# policies.UniformLoRAPolicy(6, 6, 32, True, 6),
# policies.UniformLoRAPolicy(4, 4, 32, True, 4),

gpt2.apply_qat_to_gpt2(
    model,
    policies.OutlierAdaptivePolicy(),
)

model.load_state_dict(
    safetensors.torch.load_file("./gpt2-squad-cascade/model.safetensors"),
    strict=False,
)

squad.eval_attack(model)
