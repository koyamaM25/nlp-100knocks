from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import torch
import torch.nn.functional as F
from datasets import load_dataset
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

OUTPUT_PATH = "/home/koyama/nlp-100knocks/10章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OUT_DIR = os.path.join(OUTPUT_PATH, "out_98_ft_gpt2")
OUT_TXT = os.path.join(OUTPUT_PATH, "out_98.txt")

MODEL_NAME = "gpt2"
MAX_LEN = 192
EPOCHS = 1
BATCH_SIZE = 8
LR = 5e-5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU0のみ使用
torch.cuda.set_device(0)

ds = load_dataset("glue", "sst2")
train_ds = ds["train"]
dev_ds = ds["validation"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

# Label strings (leading space is important for GPT-2 BPE)
LABEL_TEXT = {0: " negative", 1: " positive"}

def make_prompt(sentence: str) -> str:
    # Keep prompt fixed for fair eval
    return (
        "Classify the sentiment of the movie review as Positive or Negative.\n"
        f"Review: {sentence}\n"
        "Sentiment:"
    )

def encode_supervised(sentence: str, label: int) -> Dict[str, Any]:
    prompt = make_prompt(sentence)
    completion = LABEL_TEXT[int(label)]

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    comp_ids = tokenizer(completion, add_special_tokens=False).input_ids

    input_ids = prompt_ids + comp_ids
    # Only compute loss on the completion tokens
    labels = [-100] * len(prompt_ids) + comp_ids

    # Truncate (keep the END of the prompt+label if too long)
    if len(input_ids) > MAX_LEN:
        input_ids = input_ids[-MAX_LEN:]
        labels = labels[-MAX_LEN:]

        if all(x == -100 for x in labels):
            labels[-1] = input_ids[-1]

    return {"input_ids": input_ids, "labels": labels}

def map_fn(ex):
    return encode_supervised(ex["sentence"], ex["label"])

train_tok = train_ds.map(map_fn, remove_columns=train_ds.column_names)
dev_tok = dev_ds.map(map_fn, remove_columns=dev_ds.column_names)

# Data collator (pad input_ids and labels)
@dataclass
class CausalLMSupervisedCollator:
    tokenizer: AutoTokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]

        # Pad input_ids with pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # Pad labels with -100 (ignored by loss)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

collator = CausalLMSupervisedCollator(tokenizer)

# Evaluation: label log-likelihood comparison 
@torch.no_grad()
def label_logprob(model, prompt_text: str, completion_text: str) -> float:
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    comp_ids = tokenizer(completion_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)

    logits = model(input_ids=input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)

    N = prompt_ids.shape[1]
    M = comp_ids.shape[1]
    total = 0.0
    for i in range(M):
        pos = N + i
        tid = input_ids[0, pos].item()
        total += log_probs[0, pos - 1, tid].item()
    return total

@torch.no_grad()
def eval_accuracy(model, split, max_eval: int = None) -> float:
    model.eval()
    correct = 0
    total = 0
    for idx, ex in enumerate(split):
        if max_eval is not None and idx >= max_eval:
            break
        sent = ex["sentence"]
        gold = int(ex["label"])
        prompt = make_prompt(sent)

        lp_neg = label_logprob(model, prompt, LABEL_TEXT[0])
        lp_pos = label_logprob(model, prompt, LABEL_TEXT[1])
        pred = 1 if lp_pos > lp_neg else 0

        correct += int(pred == gold)
        total += 1
    return correct / total if total else 0.0

# 訓練
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    logging_steps=100,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    data_collator=collator,
)

with open(OUT_TXT, "w", encoding="UTF-8") as f:
    f.write(f"MODEL={MODEL_NAME}\n")
    f.write("TASK=GLUE SST-2\n")
    f.write("FORMAT=prompt + label_text; loss on label tokens only\n")
    f.write(f"MAX_LEN={MAX_LEN}, EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LR}\n\n")

    # Baseline (before fine-tuning)
    base_acc = eval_accuracy(model, dev_ds, max_eval=500)  # quick sanity eval
    f.write(f"dev_acc_before_ft (first 500 ex): {base_acc:.6f}\n")

trainer.train()

# Save final model
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

# Evaluate after fine-tuning (full dev)
ft_acc = eval_accuracy(model, dev_ds, max_eval=None)

with open(OUT_TXT, "a", encoding="UTF-8") as f:
    f.write(f"dev_acc_after_ft (full dev): {ft_acc:.6f}\n")

print(f"Wrote: {OUT_TXT}")
print(f"Saved fine-tuned model to: {OUT_DIR}")