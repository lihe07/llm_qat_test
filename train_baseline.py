import gpt2
from transformers import (
    GPT2ForQuestionAnswering,
    GPT2TokenizerFast,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import utils

# Load tokenizer and model
model = GPT2ForQuestionAnswering.from_pretrained("gpt2", device_map="auto")

# Add a padding token
model.resize_token_embeddings(len(utils.tokenizer))

# Prevent KV cache usage for simplicity
model.config.use_cache = False

# Load SQuAD dataset
dataset = load_dataset("squad")


tokenized_squad = dataset.map(
    utils.preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# Training
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
    tokenizer=utils.tokenizer,
)

trainer.train()

# Save model
trainer.save_model("./gpt2-squad-finetuned")
