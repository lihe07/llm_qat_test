import torch
import torch.nn as nn
import gpt2
from transformers import GPT2ForQuestionAnswering
from typing import List, Dict
import trainer


class Cascade(trainer.MyTrainer):
    """
    Implements avg_loss bit_schedule
    https://github.com/GATECH-EIC/InstantNet/blob/main/config_train.py
    """

    def __init__(
        self,
        model: GPT2ForQuestionAnswering,
        policies: List[gpt2.Policy],
        dist_weight=1.0,
    ):
        super().__init__(model, policies)
        self.hard_loss = nn.CrossEntropyLoss()

        self.dist_loss = nn.KLDivLoss(reduction="batchmean")
        # Or MSELoss as in original paper

        self.dist_weight = dist_weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        start_positions: torch.LongTensor,
        end_positions: torch.LongTensor,
    ):
        outputs: List[Dict[str, torch.Tensor]] = []

        total_loss = torch.tensor(0.0).to(input_ids.device)
        for i, policy in enumerate(self.policy_dicts):
            teacher_outputs = outputs[0:i]
            gpt2.apply_qat_to_gpt2(self.model, policy=policy)
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            output_no_grad = {
                "start_logits": output.start_logits.detach(),
                "end_logits": output.end_logits.detach(),
            }

            outputs.append(
                output_no_grad
            )  # We don't want gradients from teacher outputs

            loss = (
                self.hard_loss(output.start_logits, start_positions)
                + self.hard_loss(output.end_logits, end_positions)
            ) * 0.5

            if len(teacher_outputs) == 0:
                del output
                loss.backward()
                continue

            distill_loss = torch.tensor(0.0).to(input_ids.device)
            for teacher_output in teacher_outputs:
                if isinstance(self.dist_loss, nn.KLDivLoss):
                    distill_loss += (
                        self.dist_loss(
                            nn.functional.log_softmax(output.start_logits, dim=-1),
                            nn.functional.softmax(
                                teacher_output["start_logits"], dim=-1
                            ),
                        )
                        + self.dist_loss(
                            nn.functional.log_softmax(output.end_logits, dim=-1),
                            nn.functional.softmax(teacher_output["end_logits"], dim=-1),
                        )
                    ) * 0.5
                else:
                    distill_loss += (
                        self.dist_loss(
                            output.start_logits, teacher_output["start_logits"]
                        )
                        + self.dist_loss(
                            output.end_logits, teacher_output["end_logits"]
                        )
                    ) * 0.5

            loss += distill_loss * self.dist_weight / len(teacher_outputs)

            loss.backward()
            total_loss += loss.item()
            del loss
            del output

        total_loss.requires_grad = True  # some hack to make Trainer happy
        return (total_loss,)
