from typing import List
import gpt2
from torch import nn
import torch
from transformers import (
    GPT2ForQuestionAnswering,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import utils
import policies
import squad


class Cascade(nn.Module):
    """
    Implements avg_loss bit_schedule
    https://github.com/GATECH-EIC/InstantNet/blob/main/config_train.py
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        # convert to nn.ModuleList
        self.models = nn.ModuleList(models)
        self.hard_loss = nn.CrossEntropyLoss()

        self.dist_loss = nn.KLDivLoss(reduction="batchmean")
        # Or MSELoss as in original paper

        self.distill_weight = 1.0

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        start_positions: torch.LongTensor,
        end_positions: torch.LongTensor,
    ):
        outputs = [
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            for model in self.models
        ]
        total_loss = torch.tensor(0.0).to(input_ids.device)

        for i, output in enumerate(outputs):
            teacher_outputs = outputs[0:i]

            loss = (
                self.hard_loss(output.start_logits, start_positions)
                + self.hard_loss(output.end_logits, end_positions)
            ) * 0.5

            if len(teacher_outputs) == 0:
                total_loss += loss
                continue

            distill_loss = torch.tensor(0.0).to(input_ids.device)
            for teacher_output in teacher_outputs:
                if isinstance(self.dist_loss, nn.KLDivLoss):
                    distill_loss += (
                        self.dist_loss(
                            nn.functional.log_softmax(output.start_logits, dim=-1),
                            nn.functional.softmax(teacher_output.start_logits, dim=-1),
                        )
                        + self.dist_loss(
                            nn.functional.log_softmax(output.end_logits, dim=-1),
                            nn.functional.softmax(teacher_output.end_logits, dim=-1),
                        )
                    ) * 0.5
                else:
                    distill_loss += (
                        self.dist_loss(output.start_logits, teacher_output.start_logits)
                        + self.dist_loss(output.end_logits, teacher_output.end_logits)
                    ) * 0.5

            loss += distill_loss * self.distill_weight / len(teacher_outputs)

            total_loss += loss

        return (total_loss,)

    def get_model(self, index: int) -> PreTrainedModel:
        return self.models[index]  # type:ignore


# Load tokenizer and model
model = GPT2ForQuestionAnswering.from_pretrained(
    "./gpt2-squad-finetuned/", device_map="auto"
)

quant: GPT2ForQuestionAnswering = gpt2.shallow_clone(model)  # type:ignore
gpt2.apply_qat_to_gpt2(
    quant,
    # policy=policies.DepthAdaptivePolicy(total_layers=12),
    policy=policies.ConservativeFullBiasPolicy(),
)

# Load SQuAD dataset
dataset = load_dataset("squad")

tokenized_squad = dataset.map(
    utils.preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# Training
training_args = TrainingArguments(
    eval_strategy="no",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    max_steps=1000,
    weight_decay=0.01,
)


cascade = Cascade(
    [
        model,
        quant,
    ]
)

print("Before QAT:")
# squad.eval(quant, num=100)

trainer = Trainer(
    model=cascade,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
    processing_class=utils.tokenizer,
)

trainer.train()

print("After QAT:")
squad.eval(quant, num=1000)
