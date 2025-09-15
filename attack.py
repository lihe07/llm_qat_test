import utils
import torch
from transformers import GPT2ForQuestionAnswering
from datasets import load_dataset
from typing import Any
from rich.progress import track
from rich import print

NUM = 1000

dataset: Any = load_dataset("squad")

original_squad: Any = load_dataset("squad")

utils.max_length = 0
utils.doc_stride = 0
utils.return_overflowing_tokens = False

tokenized_squad: Any = dataset.map(
    utils.preprocess_no_truncate,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

tokenized_squad["validation"] = tokenized_squad["validation"].select(range(NUM))
original_squad["validation"] = original_squad["validation"].select(range(NUM))

model = GPT2ForQuestionAnswering.from_pretrained(
    "./gpt2-squad-finetuned/", device_map="auto"
)
model.eval()
embeddings_layer = model.transformer.wte


def find_best_flip_for_position(input_ids, embedding_gradients, embeddings_layer, i):
    # We'll assume a batch size of 1.
    gradient_at_i = embedding_gradients[0, i]

    original_token_id = input_ids[0, i]
    original_embedding = embeddings_layer(original_token_id)

    all_vocab_embeddings = embeddings_layer.weight

    dot_products = torch.matmul(
        (all_vocab_embeddings - original_embedding), gradient_at_i
    )

    # _, best_flip_token_ids = torch.topk(dot_products, 5, largest=True)
    # Sort instead of topk to get all tokens in order
    _, best_flip_token_ids = torch.sort(dot_products, descending=True)

    best_flip_id = best_flip_token_ids[0]

    i = 0
    while best_flip_id == original_token_id or best_flip_id.item() in [
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    ]:
        i += 1
        best_flip_id = best_flip_token_ids[i]

    return best_flip_id.item()


new_input_ids = []

for j, q in enumerate(track(tokenized_squad["validation"])):
    input_ids = q["input_ids"]
    input_ids = torch.tensor([input_ids])
    input_ids = input_ids.to(model.device)
    input_embeddings = embeddings_layer(input_ids)
    input_embeddings.retain_grad()
    attention_mask = torch.tensor([q["attention_mask"]]).to(model.device)
    start_positions = torch.tensor([q["start_positions"]]).to(model.device)
    end_positions = torch.tensor([q["end_positions"]]).to(model.device)

    outputs = model(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
        start_positions=start_positions,
        end_positions=end_positions,
        return_dict=True,
    )

    outputs.loss.backward()

    embedding_gradients = input_embeddings.grad

    original_loss = outputs.loss

    max_loss = -1
    best_adversarial_ids = None
    sequence_length = input_ids.shape[1]

    print(f"Original Loss: {original_loss.item():.4f}")

    tokenizer = utils.tokenizer

    sep_location = len(tokenizer(original_squad["validation"][j]["question"]).input_ids)

    print(f"SEP at {sep_location}")
    for i in range(sequence_length):
        # Don't try to flip special tokens like [CLS], [SEP], [PAD]
        if input_ids[0, i] in [
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        ]:
            continue
        # Don't try to flip the question tokens (everything before the first SEP)
        if i <= sep_location + 1:
            continue

        # Don't try to flip tokens that are near the answer
        if start_positions - 3 <= i <= end_positions + 3:
            continue

        replacement_id = find_best_flip_for_position(
            input_ids, embedding_gradients, embeddings_layer, i
        )

        adversarial_ids_candidate = input_ids.clone()
        adversarial_ids_candidate[0, i] = replacement_id

        outputs = model(
            input_ids=adversarial_ids_candidate,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        current_loss = outputs.loss

        # If this flip is the best so far, save it
        if current_loss > max_loss:
            max_loss = current_loss
            best_adversarial_ids = adversarial_ids_candidate
            print(
                f"  New best attack: Pos {i}, '{tokenizer.decode(input_ids[0,i])}' -> '{tokenizer.decode(replacement_id)}', Loss: {current_loss.item():.4f}"
            )

    new_input_ids.append(best_adversarial_ids[0].cpu().numpy().tolist())


original_squad = original_squad["validation"].add_column("input_ids", new_input_ids)

print("Done")
original_squad.save_to_disk("./squad-adversarial")
