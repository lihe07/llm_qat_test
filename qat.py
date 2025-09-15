from typing import Any
import gpt2
from transformers import (
    GPT2ForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import utils
import policies
from cascade import Cascade
from cyclic import Cyclic


# Load tokenizer and model
model = GPT2ForQuestionAnswering.from_pretrained(
    "./gpt2-squad-finetuned/", device_map="auto"
)

# Load SQuAD dataset
dataset: Any = load_dataset("squad")

tokenized_squad: Any = dataset.map(
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
    max_steps=1000,
    weight_decay=0.01,
)

policies_list = [
    gpt2.PlaceboPolicy(),
    policies.OutlierAdaptivePolicy(),
    policies.AggressiveLowBitLoRA(total_layers=12),
    policies.DepthAdaptivePolicy(total_layers=12),
    policies.ConservativeLoRAPolicy(full_bias=True),
    policies.UniformLoRAPolicy(8, 8, 32, True, 8),
    policies.UniformLoRAPolicy(6, 6, 32, True, 6),
    policies.UniformLoRAPolicy(4, 4, 32, True, 4),
    # policies.UniformLoRAPolicy(8, 8, 32, False, 8),
    # policies.UniformLoRAPolicy(7, 7, 32, False, 7),
    # policies.UniformLoRAPolicy(6, 6, 32, False, 6),
    # policies.UniformLoRAPolicy(5, 5, 32, False, 5),
    # policies.UniformLoRAPolicy(4, 4, 32, False, 4),
]


my_trainer = Cascade(model, policies_list)
# my_trainer = Cyclic(model, policies_list)

print("Using Trainer:", my_trainer.__class__.__name__)

print("Before QAT:")
# my_trainer.eval_all()

trainer = Trainer(
    model=my_trainer,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
    processing_class=utils.tokenizer,
)

trainer.train()

print("After QAT:")

# my_trainer.eval_all()

my_trainer.save("./gpt2-squad-cascade")
