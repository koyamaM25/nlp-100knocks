import pickle
import random
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# =========================
# パス設定（必要に応じて変更）
# =========================
EMB_MATRIX_PATH = "/home/koyama/nlp-100knocks/8章/out/out_embedding_matrix_70.npy"
TRAIN_PKL_PATH  = "/home/koyama/nlp-100knocks/8章/out/out_train_71.pkl"
DEV_PKL_PATH    = "/home/koyama/nlp-100knocks/8章/out/out_dev_71.pkl"

PAD_ID = 0  # 問題70で <PAD>=0

# =========================
# 学習設定
# =========================
SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 0.0
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 乱数固定
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Dataset（問題71の形式に合わせる）
# =========================
class SSTDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        return {
            "input_ids": item["input_ids"],  # 1D LongTensor
            "label": item["label"],          # FloatTensor shape (1,)
        }


# =========================
# collate_fn（可変長PAD）
# =========================
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_ids_list = [x["input_ids"] for x in batch]
    labels_list = [x["label"] for x in batch]

    lengths = torch.tensor([t.size(0) for t in input_ids_list], dtype=torch.long)

    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=PAD_ID
    )  # (B, Lmax)

    attention_mask = (input_ids != PAD_ID).long()  # (B, Lmax)
    labels = torch.stack(labels_list, dim=0).float()  # (B, 1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "lengths": lengths,
    }


# =========================
# モデル：Embedding + mean pooling + Linear
# =========================
class MeanPoolClassifier(nn.Module):
    def __init__(self, emb_matrix: np.ndarray, pad_id: int = 0, freeze_emb: bool = False):
        super().__init__()
        emb_tensor = torch.from_numpy(emb_matrix)  # float32 (V, D)

        self.embedding = nn.Embedding.from_pretrained(
            embeddings=emb_tensor,
            freeze=freeze_emb,      # ★問題74：ここを比較
            padding_idx=pad_id
        )
        emb_dim = emb_tensor.size(1)
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # (B, L, D)

        # PADを除いて平均（mean pooling）
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        x = x * mask
        sum_vec = x.sum(dim=1)                  # (B, D)
        denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        mean_vec = sum_vec / denom              # (B, D)

        logits = self.classifier(mean_vec)      # (B, 1)
        return logits


# =========================
# 評価（loss / accuracy）
# =========================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)  # (B,1)

        logits = model(input_ids, attention_mask)  # (B,1)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_count += labels.numel()

    return {
        "loss": total_loss / max(1, total_count),
        "acc": total_correct / max(1, total_count),
    }


# =========================
# 学習（freezeの設定を変えて比較）
# =========================
def train_and_eval(freeze_emb: bool, emb_matrix: np.ndarray, train_loader: DataLoader, dev_loader: DataLoader) -> Dict[str, float]:
    model = MeanPoolClassifier(emb_matrix, pad_id=PAD_ID, freeze_emb=freeze_emb).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_dev_acc = 0.0
    best_dev_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / max(1, total)
        dev_metrics = evaluate(model, dev_loader, criterion)

        best_dev_acc = max(best_dev_acc, dev_metrics["acc"])
        best_dev_loss = min(best_dev_loss, dev_metrics["loss"])

        mode = "freeze" if freeze_emb else "finetune"
        print(
            f"[{mode}][Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} | dev_loss={dev_metrics['loss']:.4f} | dev_acc={dev_metrics['acc']:.4f}"
        )

    return {
        "freeze_emb": freeze_emb,
        "best_dev_acc": best_dev_acc,
        "best_dev_loss": best_dev_loss,
    }


# =========================
# main
# =========================
def main():
    set_seed(SEED)
    print(f"device: {DEVICE}")

    emb_matrix = np.load(EMB_MATRIX_PATH)
    print("embedding matrix:", emb_matrix.shape)

    with open(TRAIN_PKL_PATH, "rb") as f:
        train_data = pickle.load(f)
    with open(DEV_PKL_PATH, "rb") as f:
        dev_data = pickle.load(f)

    train_loader = DataLoader(
        SSTDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        SSTDataset(dev_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    print("\n" + "=" * 60)
    print("Case 1: 埋め込みを固定（freeze_emb=True）")
    print("=" * 60)
    res_freeze = train_and_eval(True, emb_matrix, train_loader, dev_loader)

    print("\n" + "=" * 60)
    print("Case 2: 埋め込みも更新（freeze_emb=False）")
    print("=" * 60)
    res_finetune = train_and_eval(False, emb_matrix, train_loader, dev_loader)

    print("\n=== Summary ===")
    for r in [res_freeze, res_finetune]:
        mode = "freeze" if r["freeze_emb"] else "finetune"
        print(f"{mode:8s} | best_dev_acc={r['best_dev_acc']:.4f} | best_dev_loss={r['best_dev_loss']:.4f}")


if __name__ == "__main__":
    main()
