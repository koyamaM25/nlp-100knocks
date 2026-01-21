import os
import pickle
import random
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

EMB_MATRIX_PATH = "/home/koyama/nlp-100knocks/8章/out/out_70_embedding_matrix.npy"
TRAIN_PKL_PATH  = "/home/koyama/nlp-100knocks/8章/out/out_71_train.pkl"
# DEV_PKL_PATH    = "/home/koyama/nlp-100knocks/8章/out/out_71_dev.pkl"

OUT_DIR = "/home/koyama/nlp-100knocks/8章/out"
os.makedirs(OUT_DIR, exist_ok=True)

PAD_ID = 0

# 学習設定
SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 0.0
NUM_WORKERS = 0

# 埋め込みは固定（問73の要件）
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

# Dataset
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

# 可変長をPADしてバッチ化
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_ids_list = [x["input_ids"] for x in batch]
    labels_list = [x["label"] for x in batch]

    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=PAD_ID
    )  # (B, Lmax)

    # labels: (B,) に整形
    labels = torch.stack([t.reshape(()) for t in labels_list], dim=0).float()  # (B,)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }

# Embedding + mean pooling + Linear（logits出力）
class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, emb_matrix: np.ndarray, freeze: bool = True):
        super().__init__()
        emb_tensor = torch.tensor(emb_matrix, dtype=torch.float32)

        self.embedding = nn.Embedding.from_pretrained(
            emb_tensor,
            freeze=freeze,
            padding_idx=PAD_ID,
        )
        emb_dim = emb_tensor.shape[1]
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # (B, L, D)

        # PAD(=0) を除外して平均
        mask = (input_ids != PAD_ID).unsqueeze(-1).float()  # (B, L, 1)
        x = x * mask

        lengths = mask.sum(dim=1).clamp(min=1.0)            # (B, 1)
        sent_vec = x.sum(dim=1) / lengths                   # (B, D)

        logits = self.fc(sent_vec).squeeze(-1)              # (B,)
        return logits

def main():
    set_seed(SEED)
    print(f"device: {DEVICE}")

    # 埋め込み行列（問題70の出力）
    emb_matrix = np.load(EMB_MATRIX_PATH)  # (V, D) float32
    print("embedding matrix:", emb_matrix.shape)

    # trainデータのみロード（学習で終わらせる）
    with open(TRAIN_PKL_PATH, "rb") as f:
        train_data = pickle.load(f)

    train_dataset = SSTDataset(train_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # モデル
    model = MeanEmbeddingClassifier(emb_matrix, freeze=FREEZE_EMB).to(DEVICE)

    # 損失
    criterion = nn.BCEWithLogitsLoss()

    # 最適化（問73の意図：線形層のみ学習）
    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # 学習（各epoch終了時に保存）
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

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}")

        # fcだけ保存
        epoch_save_path = os.path.join(OUT_DIR, f"out_73_fc_epoch{epoch:02d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "fc_state_dict": model.fc.state_dict(),  # ←ここだけ！
                "pad_id": PAD_ID,
                "freeze_emb": FREEZE_EMB,
                "emb_matrix_path": EMB_MATRIX_PATH,
            },
            epoch_save_path
        )
        print(f"saved: {epoch_save_path}")


    print("training finished.")

if __name__ == "__main__":
    main()

