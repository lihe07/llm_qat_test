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


def squad(
    model: GPT2ForQuestionAnswering, context, question, tokenizer: GPT2TokenizerFast
):
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
            return "", -1

        answer_ids = inputs.input_ids[0, start_index : end_index + 1]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
        # calculate start_char

    return answer, inputs.token_to_chars(int(start_index.item())).start
