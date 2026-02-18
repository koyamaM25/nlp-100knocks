from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import zipfile
import os
from typing import Dict, Any, List

# =========================
# Paths / Config
# =========================
OUTPUT_PATH = "/home/koyama/nlp-100knocks/10章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OUT_LOG = os.path.join(OUTPUT_PATH, "out_97.txt")
OUT_DIR = os.path.join(OUTPUT_PATH, "out_97_trainer_ckpt")
os.makedirs(OUT_DIR, exist_ok=True)

SST2_PATH = "/home/koyama/nlp-100knocks/7章/code/SST-2.zip"

MODEL_NAME = "gpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 128
TRAIN_BS = 32
EVAL_BS = 64
EPOCHS = 3
LR = 1e-3
SEED = 42

# =========================
# Load SST-2 from zip
# =========================
def load_sst2_data():
    with zipfile.ZipFile(SST2_PATH, "r") as z:
        with z.open("SST-2/train.tsv") as f:
            df_train = pd.read_csv(f, sep="\t")
        with z.open("SST-2/dev.tsv") as f:
            df_dev = pd.read_csv(f, sep="\t")
    return df_train, df_dev

df_train, df_dev = load_sst2_data()

# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# GPT-2 has no pad token by default -> set pad to eos for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# Torch Dataset (pre-tokenized)
# =========================
class SST2TorchDataset(torch.utils.data.Dataset):
    def __init__(self, sentences: List[str], labels: List[int], tokenizer, max_len: int):
        self.enc = tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": self.labels[idx],
        }

train_dataset = SST2TorchDataset(
    df_train["sentence"].tolist(),
    df_train["label"].astype(int).tolist(),
    tokenizer,
    MAX_LEN,
)
dev_dataset = SST2TorchDataset(
    df_dev["sentence"].tolist(),
    df_dev["label"].astype(int).tolist(),
    tokenizer,
    MAX_LEN,
)

# =========================
# Model: Frozen GPT-2 encoder + FFN classifier
# =========================
class GPT2EmbedClassifier(nn.Module):
    def __init__(self, model_name: str, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)  # GPT2Model
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden_size = self.encoder.config.n_embd
        self.fc1 = nn.Linear(hidden_size, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def masked_mean_pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden: (B, L, H), attention_mask: (B, L)
        mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
        summed = (last_hidden * mask).sum(dim=1)             # (B, H)
        counts = mask.sum(dim=1).clamp(min=1.0)              # (B, 1)
        return summed / counts                               # (B, H)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = out.last_hidden_state                  # (B, L, H)
        emb = self.masked_mean_pool(last_hidden, attention_mask)  # (B, H)

        x = F.relu(self.fc1(emb))
        x = self.drop(x)
        logits = self.fc2(x)                                 # (B, 2)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        # Trainer互換の返り値（loss, logits）
        return {"loss": loss, "logits": logits}

model = GPT2EmbedClassifier(MODEL_NAME).to(DEVICE)

# TrainingArguments
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    learning_rate=LR,
    logging_steps=100,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    # tokenizer=tokenizer,  # 古いtransformersだと未対応なので渡さない
)

trainer.train()

# =========================
# Evaluate (accuracy)
# =========================
pred_out = trainer.predict(dev_dataset)
preds = pred_out.predictions
# predictions can be numpy or torch; normalize
if isinstance(preds, torch.Tensor):
    pred_labels = preds.argmax(dim=-1).cpu().numpy()
else:
    import numpy as np
    pred_labels = preds.argmax(axis=-1)

gold = df_dev["label"].astype(int).to_numpy()
acc = (pred_labels == gold).mean()

with open(OUT_LOG, "w", encoding="utf-8") as f:
    f.write("Problem 97: Embedding-based Sentiment Analysis (Trainer)\n")
    f.write(f"MODEL={MODEL_NAME}\n")
    f.write("ENCODER=frozen GPT-2 (last_hidden masked mean pooling)\n")
    f.write("HEAD=FFN (Linear-ReLU-Dropout-Linear)\n")
    f.write(f"MAX_LEN={MAX_LEN}, TRAIN_BS={TRAIN_BS}, EVAL_BS={EVAL_BS}, EPOCHS={EPOCHS}, LR={LR}\n")
    f.write(f"dev_accuracy={acc:.6f}\n")

print(f"Wrote: {OUT_LOG}")
print(f"dev_accuracy={acc:.6f}")