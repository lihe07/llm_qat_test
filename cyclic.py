import math
import torch
import torch.nn as nn
import gpt2
from transformers import GPT2ForQuestionAnswering
from typing import List
import trainer


class Cyclic(trainer.MyTrainer):
    """
    Implements Cyclic Precision Training
    """

    def __init__(
        self,
        model: GPT2ForQuestionAnswering,
        policies: List[gpt2.Policy],
        steps_per_cycle: int = 100,
    ):
        super().__init__(model, policies)
        self.loss = nn.CrossEntropyLoss()
        self.steps_per_cycle = steps_per_cycle
        self.step_in_cycle = 0

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        start_positions: torch.LongTensor,
        end_positions: torch.LongTensor,
    ):
        idx = math.floor(
            (self.step_in_cycle / self.steps_per_cycle) * len(self.policy_dicts)
        )
        idx = min(idx, len(self.policy_dicts) - 1)

        print(f"Using policy {idx} in cycle step {self.step_in_cycle}")

        current_policy = self.policy_dicts[idx]

        gpt2.apply_qat_to_gpt2(self.model, policy=current_policy)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=True,
        )

        self.step_in_cycle += 1
        self.step_in_cycle = self.step_in_cycle % self.steps_per_cycle

        return output
