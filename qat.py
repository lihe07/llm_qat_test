from typing import List
from transformers.utils.generic import ModelOutput
import gpt2
from torch import nn
import torch
from transformers import (
    GPT2ForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import utils
import policies
import squad

# Load tokenizer and model
model = GPT2ForQuestionAnswering.from_pretrained(
    "./gpt2-squad-finetuned/", device_map="auto"
)

conservative = gpt2.shallow_clone(model)
gpt2.apply_qat_to_gpt2(
    conservative,
    policy=policies.ConservativeStablePolicy(),
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
    output_dir="./results-qat-conservative",
    eval_strategy="no",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    max_steps=1000,
    weight_decay=0.01,
)


class Cascade(nn.Module):
    """
    Implements avg_loss bit_schedule
    https://github.com/GATECH-EIC/InstantNet/blob/main/config_train.py
    """

    def __init__(self, models: List[GPT2ForQuestionAnswering]):
        super().__init__()
        self.model = model
        self.conservative = conservative
        self.loss_fct = nn.CrossEntropyLoss()
        self.distill_weight = 1

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        start_positions: torch.LongTensor,
        end_positions: torch.LongTensor,
    ):
        y1 = self.conservative(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=True,
        )

        return y1

        y2 = self.conservative(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        loss1 = y1.loss
        loss2 = (
            self.loss_fct(y2.start_logits, y1.start_logits)
            + self.loss_fct(y2.end_logits, y1.end_logits)
        ) * 0.5

        loss = loss1 + loss2

        return (loss,)


cascade = Cascade()

trainer = Trainer(
    model=cascade,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
    processing_class=utils.tokenizer,
)

trainer.train()

# Save model
trainer.model.conservative.save_pretrained("./gpt2-qat-conservative")
