import os
import pickle
import random
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

EMB_MATRIX_PATH = "/home/koyama/nlp-100knocks/8章/out/out_70_embedding_matrix.npy"
TRAIN_PKL_PATH  = "/home/koyama/nlp-100knocks/8章/out/out_71_train.pkl"
DEV_PKL_PATH    = "/home/koyama/nlp-100knocks/8章/out/out_71_dev.pkl"

OUT_DIR = "/home/koyama/nlp-100knocks/8章/out"
os.makedirs(OUT_DIR, exist_ok=True)

SAVE_PATH = os.path.join(OUT_DIR, "out_73_model.pt")

PAD_ID = 0

# 学習設定
SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 0.0
NUM_WORKERS = 0

# 埋め込みは固定
FREEZE_EMB = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 乱数固定
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dataset（問題71の形式に合わせる）
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
# collate_fn（可変長をPADしてバッチ化）
# =========================
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_ids_list = [x["input_ids"] for x in batch]
    labels_list = [x["label"] for x in batch]

    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=PAD_ID
    )  # (B, Lmax)

    # labels: (B,) に整形（(B,1)でも動くが合わせておく）
    labels = torch.stack([t.reshape(()) for t in labels_list], dim=0).float()  # (B,)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


# =========================
# 問72モデル：Embedding + mean pooling + Linear（logits出力）
# =========================
class MeanEmbeddingClassifier(nn.Module):
    """
    問72: 単語埋め込みの平均ベクトルを文ベクトルとして用い、
          線形層で2値分類（SST-2: 0/1）するモデル。
    - embedding は問70で作った行列を使用
    - padding_idx=0 を想定
    - 出力は「ロジット」(sigmoid前) を返す
    """
    def __init__(self, emb_matrix: np.ndarray, freeze: bool = True):
        super().__init__()
        emb_tensor = torch.tensor(emb_matrix, dtype=torch.float32)

        self.embedding = nn.Embedding.from_pretrained(
            emb_tensor,
            freeze=freeze,
            padding_idx=PAD_ID,
        )
        emb_dim = emb_tensor.shape[1]
        self.fc = nn.Linear(emb_dim, 1)  # 2値なので1ユニット（BCEWithLogitsLoss想定）

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, L) の LongTensor（padding込み）
        returns: logits (B,) の FloatTensor
        """
        x = self.embedding(input_ids)  # (B, L, D)

        # PAD(=0) を除外して平均
        mask = (input_ids != PAD_ID).unsqueeze(-1).float()  # (B, L, 1)
        x = x * mask

        lengths = mask.sum(dim=1).clamp(min=1.0)            # (B, 1)
        sent_vec = x.sum(dim=1) / lengths                   # (B, D)

        logits = self.fc(sent_vec).squeeze(-1)              # (B,)
        return logits


# =========================
# 評価（loss / accuracy）
# =========================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0  # サンプル数（Bの合計）

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)  # (B,)

        logits = model(input_ids)            # (B,)
        loss = criterion(logits, labels)     # scalar（batch平均）

        bsz = labels.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        total_correct += (preds == labels).sum().item()

    return {
        "loss": total_loss / max(1, total_count),
        "acc": total_correct / max(1, total_count),
    }


# =========================
# main
# =========================
def main():
    set_seed(SEED)
    print(f"device: {DEVICE}")

    # 埋め込み行列（問題70の出力）
    emb_matrix = np.load(EMB_MATRIX_PATH)  # (V, D) float32
    print("embedding matrix:", emb_matrix.shape)

    # データ（問題71の出力）
    with open(TRAIN_PKL_PATH, "rb") as f:
        train_data = pickle.load(f)
    with open(DEV_PKL_PATH, "rb") as f:
        dev_data = pickle.load(f)

    train_dataset = SSTDataset(train_data)
    dev_dataset = SSTDataset(dev_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # モデル（問72）
    model = MeanEmbeddingClassifier(emb_matrix, freeze=FREEZE_EMB).to(DEVICE)

    # 損失：2値分類（logitsを直接入れる）
    criterion = nn.BCEWithLogitsLoss()

    # 最適化：問73の意図を明確化（線形層のみ学習）
    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # 学習
    best_dev_acc = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)  # (B,)

            optimizer.zero_grad()

            logits = model(input_ids)            # (B,)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            bsz = labels.size(0)
            running_loss += loss.item() * bsz
            total += bsz

        train_loss = running_loss / max(1, total)
        dev_metrics = evaluate(model, dev_loader, criterion)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} | "
            f"dev_loss={dev_metrics['loss']:.4f} | "
            f"dev_acc={dev_metrics['acc']:.4f}"
        )

        # ベストモデル保存（任意）
        if dev_metrics["acc"] > best_dev_acc:
            best_dev_acc = dev_metrics["acc"]
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"saved(best): {SAVE_PATH}")
    print(f"best_dev_acc: {best_dev_acc:.4f}")


if __name__ == "__main__":
    main()
