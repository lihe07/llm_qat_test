import dataclasses
import json
import squad
from transformers import GPT2ForQuestionAnswering
import gpt2
import safetensors.torch
import policies

model = GPT2ForQuestionAnswering.from_pretrained(
    "./gpt2-squad-finetuned/", device_map="auto"
)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)  # type:ignore
        return super().default(o)


p = [
    policies.UniformLoRAPolicy(4, 4, 32, True, 4),
]


policy_dicts = [gpt2.get_policy_dict(model, policy) for policy in p]


gpt2.apply_qat_to_gpt2(
    model,
    policies.UniformLoRAPolicy(8, 8, 32, True, 8),
    # policies.OutlierAdaptivePolicy(),
)


model.load_state_dict(
    safetensors.torch.load_file("./gpt2-squad-cascade/model.safetensors"),
    strict=False,
)

gpt2.apply_qat_to_gpt2(model, policy_dicts[0])

squad.eval_attack(model)
