from typing import Dict, List
from transformers import GPT2ForQuestionAnswering
from torch import nn
import gpt2
import squad


class MyTrainer(nn.Module):
    """
    Parent class for Cascade and Cyclic
    """

    def __init__(
        self,
        model: GPT2ForQuestionAnswering,
        policies: List[gpt2.Policy],
    ):
        super().__init__()
        self.model = model
        self.policies = policies
        self.prepare_policies()

    def prepare_policies(self):
        self.policy_dicts = []
        for policy in self.policies:
            d = gpt2.get_policy_dict(self.model, policy)
            # Test the policy, also calculate the bitbudget
            gpt2.apply_qat_to_gpt2(self.model, policy=d)
            print("Policy:", policy.__repr__())
            print("Bitbudget:", gpt2.calculate_bit_budget(self.model))

            self.policy_dicts.append(d)

    def eval_all(self, num=-1):
        for policy, original in zip(self.policy_dicts, self.policies):
            gpt2.apply_qat_to_gpt2(self.model, policy=policy)
            print("Policy:", original.__repr__())
            squad.eval(self.model, num)
