import torch
from transformers import (
    GPT2ForQuestionAnswering,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2TokenizerFast,
)


def generate(model: GPT2LMHeadModel, tokenizer, prompt, max_length=30, temperature=1.0):
    """
    Generates a sequence of text using a model.
    """
    # input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # init from nothing
    input_ids = torch.tensor([[tokenizer.bos_token_id]])
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs.logits

            # Get the logits for the last token and apply temperature
            next_token_logits = logits[:, -1, :] / temperature

            # Get the next token id (using argmax for simplicity)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Append the new token
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            # Check for EOS token
            if (
                tokenizer.eos_token_id is not None
                and next_token_id.item() == tokenizer.eos_token_id
            ):
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def squad(model: GPT2ForQuestionAnswering, context, question):
    inputs = tokenizer(question, context, return_tensors="pt")
    # input is <question><sep><context>

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)

        if start_index > end_index:
            return ""

        answer_ids = inputs.input_ids[0, start_index : end_index + 1]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
        # calculate start_char

    return answer


try:
    tokenizer = GPT2TokenizerFast.from_pretrained("./gpt2-squad-finetuned")
except Exception:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})


# Preprocessing
max_length = 384
doc_stride = 128


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
