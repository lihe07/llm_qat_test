import gpt2
from transformers import (
    GPT2ForQuestionAnswering,
    GPT2TokenizerFast,
)
from datasets import Dataset, load_dataset
import torch
import utils
import re
import string
import collections
from rich.progress import track


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


m = GPT2ForQuestionAnswering.from_pretrained(
    "./gpt2-squad-finetuned/", device_map="auto"
)
gpt2.apply_qat_to_gpt2(m)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

dataset = load_dataset("squad")

total_exacts = 0
total_f1s = 0
print("Size of validation set:", len(dataset["validation"]))
num = len(dataset["validation"])

for question in track(dataset["validation"].select(range(num))):
    ans, start_char = utils.squad(
        m, question["context"], question["question"], tokenizer
    )
    # Check if it is correct
    exact_score = max(compute_exact(a, ans) for a in question["answers"]["text"])
    f1_score = max(compute_f1(a, ans) for a in question["answers"]["text"])
    total_exacts += exact_score
    total_f1s += f1_score

print(f"Exact: {total_exacts/num}, F1: {total_f1s/num}")
