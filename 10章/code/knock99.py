# Problem 99 (Preference Tuning / DPO) — “正解コード” (works even on古いtransformers)
#
# 方針:
# - SST-2 を「好みデータ」に変換（正解ラベル文字列を chosen、逆ラベルを rejected）
# - DPO (Direct Preference Optimization) を手書き実装
# - policy(model) を更新し、reference(ref_model) は固定
# - 出力: out_99.txt（学習ログ + fine-tune前後のdev accuracy）
#
# 依存:
#   pip install datasets transformers torch
# (trl不要)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from datasets import load_dataset
import os
import math
from typing import Dict, Any, List

OUTPUT_PATH = "/home/koyama/nlp-100knocks/10章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OUT_DIR = os.path.join(OUTPUT_PATH, "out_99_dpo_gpt2")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_LOG = os.path.join(OUTPUT_PATH, "out_99.txt")

MODEL_NAME = "gpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 192          # prompt+completion の最大長（トークン数）
BATCH_SIZE = 4         # GPUに応じて調整
EPOCHS = 1             
LR = 1e-5
BETA = 0.1             # DPOのβ
LOG_EVERY = 50

# 学習を軽くしたい場合
MAX_TRAIN_EXAMPLES = None   
MAX_DEV_EXAMPLES = None     

ds = load_dataset("glue", "sst2")
train_ds = ds["train"]
dev_ds = ds["validation"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.train()

ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

LABEL_TEXT = {0: " negative", 1: " positive"}  

def make_prompt(sentence: str) -> str:
    return (
        "Classify the sentiment of the movie review as Positive or Negative.\n"
        f"Review: {sentence}\n"
        "Sentiment:"
    )

@torch.no_grad()
def seq_logprob(model_, prompt_text: str, completion_text: str) -> torch.Tensor:
    """
    Return log p(completion | prompt) as a scalar tensor (on DEVICE).
    completion_text can be multi-token.
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    comp_ids = tokenizer(completion_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)  

    logits = model_(input_ids=input_ids).logits 
    log_probs = F.log_softmax(logits, dim=-1)

    N = prompt_ids.shape[1]
    M = comp_ids.shape[1]

    total = torch.zeros((), device=DEVICE)
    for i in range(M):
        pos = N + i
        tid = input_ids[0, pos]
        total = total + log_probs[0, pos - 1, tid]
    return total

def seq_logprob_grad(model_, prompt_text: str, completion_text: str) -> torch.Tensor:
    """
    Same as seq_logprob but WITH grad for policy model.
    Return scalar tensor.
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    comp_ids = tokenizer(completion_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)

    logits = model_(input_ids=input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)

    N = prompt_ids.shape[1]
    M = comp_ids.shape[1]

    total = torch.zeros((), device=DEVICE)
    for i in range(M):
        pos = N + i
        tid = input_ids[0, pos]
        total = total + log_probs[0, pos - 1, tid]
    return total

def build_pref_examples(split, max_n=None) -> List[Dict[str, Any]]:
    exs = []
    for idx, ex in enumerate(split):
        if max_n is not None and idx >= max_n:
            break
        sent = ex["sentence"]
        label = int(ex["label"])
        prompt = make_prompt(sent)
        chosen = LABEL_TEXT[label]
        rejected = LABEL_TEXT[1 - label]
        exs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "gold": label, "sentence": sent})
    return exs

train_exs = build_pref_examples(train_ds, MAX_TRAIN_EXAMPLES)
dev_exs = build_pref_examples(dev_ds, MAX_DEV_EXAMPLES)

def batch_iter(data: List[Dict[str, Any]], batch_size: int, shuffle: bool = True):
    idxs = list(range(len(data)))
    if shuffle:
        g = torch.Generator()
        g.manual_seed(42)
        idxs = torch.randperm(len(data), generator=g).tolist()
    for i in range(0, len(idxs), batch_size):
        yield [data[j] for j in idxs[i:i+batch_size]]

@torch.no_grad()
def eval_accuracy(model_, data: List[Dict[str, Any]]) -> float:
    model_.eval()
    correct = 0
    total = 0
    for ex in data:
        prompt = ex["prompt"]
        gold = ex["gold"]

        lp_neg = seq_logprob(model_, prompt, LABEL_TEXT[0]).item()
        lp_pos = seq_logprob(model_, prompt, LABEL_TEXT[1]).item()
        pred = 1 if lp_pos > lp_neg else 0

        correct += int(pred == gold)
        total += 1
    model_.train()
    return correct / total if total else 0.0


# DPO訓練
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

def dpo_loss_for_batch(batch: List[Dict[str, Any]]) -> torch.Tensor:
    """
    DPO loss:
      loss = -log sigmoid( beta * ( (logπ(yc)-logπ(yr)) - (logπref(yc)-logπref(yr)) ) )
    """
    losses = []
    for ex in batch:
        prompt = ex["prompt"]
        yc = ex["chosen"]
        yr = ex["rejected"]

        # policy logprobs (with grad)
        lp_pi_c = seq_logprob_grad(model, prompt, yc)
        lp_pi_r = seq_logprob_grad(model, prompt, yr)

        # reference logprobs (no grad)
        with torch.no_grad():
            lp_ref_c = seq_logprob(ref_model, prompt, yc)
            lp_ref_r = seq_logprob(ref_model, prompt, yr)

        adv = (lp_pi_c - lp_pi_r) - (lp_ref_c - lp_ref_r)
        losses.append(-F.logsigmoid(BETA * adv))
    return torch.stack(losses).mean()

# 実行
with open(OUT_LOG, "w", encoding="UTF-8") as f:
    f.write(f"MODEL={MODEL_NAME}\n")
    f.write("METHOD=DPO (preference tuning)\n")
    f.write("DATA=SST-2 converted to (prompt, chosen, rejected)\n")
    f.write(f"MAX_LEN={MAX_LEN} (note: this script uses per-example concat; MAX_LEN not enforced)\n")
    f.write(f"EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LR}, BETA={BETA}\n\n")

    # baseline
    base_acc = eval_accuracy(model, dev_exs[:500] if len(dev_exs) > 500 else dev_exs)
    f.write(f"dev_acc_before (up to 500 ex): {base_acc:.6f}\n\n")
    f.flush()

    step = 0
    for epoch in range(1, EPOCHS + 1):
        running = 0.0
        n = 0
        for batch in batch_iter(train_exs, BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            loss = dpo_loss_for_batch(batch)
            loss.backward()
            optimizer.step()

            step += 1
            running += loss.item()
            n += 1

            if step % LOG_EVERY == 0:
                avg_loss = running / max(n, 1)
                f.write(f"epoch={epoch}\tstep={step}\ttrain_loss={avg_loss:.6f}\n")
                f.flush()
                running, n = 0.0, 0

        # epoch end eval
        dev_acc = eval_accuracy(model, dev_exs[:500] if len(dev_exs) > 500 else dev_exs)
        f.write(f"epoch_end={epoch}\tdev_acc (up to 500 ex): {dev_acc:.6f}\n\n")
        f.flush()

# 保存
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print(f"Wrote: {OUT_LOG}")
print(f"Saved DPO-tuned model to: {OUT_DIR}")