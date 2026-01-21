import os
import pickle
import random
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# =========================
# パス設定（必要に応じて変更）
# =========================
EMB_MATRIX_PATH = "/home/koyama/nlp-100knocks/8章/out/out_70_embedding_matrix.npy"
TRAIN_PKL_PATH  = "/home/koyama/nlp-100knocks/8章/out/out_71_train.pkl"
DEV_PKL_PATH    = "/home/koyama/nlp-100knocks/8章/out/out_71_dev.pkl"

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

# BiLSTM設定
HIDDEN_DIM = 128
NUM_LAYERS = 1
BIDIRECTIONAL = True  # ★問題77：双方向
LSTM_DROPOUT = 0.0    # num_layers>=2 なら 0.3 などにしてもよい

# 分類器側Dropout（必要なら）
CLS_DROPOUT = 0.2

FREEZE_EMB = False

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
            "input_ids": item["input_ids"],  # 1D LongTensor（可変長）
            "label": item["label"],          # FloatTensor shape (1,)
        }


# =========================
# collate_fn（可変長PAD + lengths）
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

    attention_mask = (input_ids != PAD_ID).long()
    labels = torch.stack(labels_list, dim=0).float()  # (B, 1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "lengths": lengths,
    }


# =========================
# モデル：Embedding + BiLSTM + Linear
# =========================
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        emb_matrix: np.ndarray,
        pad_id: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 1,
        lstm_dropout: float = 0.0,
        cls_dropout: float = 0.2,
        freeze_emb: bool = False,
    ):
        super().__init__()

        emb_tensor = torch.from_numpy(emb_matrix)  # float32 (V, D)
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=emb_tensor,
            freeze=freeze_emb,
            padding_idx=pad_id
        )
        emb_dim = emb_tensor.size(1)

        self.num_layers = num_layers
        self.bidirectional = True
        self.num_directions = 2

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers >= 2 else 0.0
        )

        # 最終層の forward/backward を結合するので 2*hidden_dim
        out_dim = hidden_dim * self.num_directions

        self.dropout = nn.Dropout(p=cls_dropout)
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, L)
        lengths  : (B,)
        return   : logits (B, 1)
        """
        x = self.embedding(input_ids)  # (B, L, D)

        # PADを無視してLSTM（pack）
        packed = pack_padded_sequence(
            x,
            lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False
        )

        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * num_directions, B, H)

        # 最終層の forward/backward を取り出して結合
        # forward: -2, backward: -1
        h_forward = h_n[-2]  # (B, H)
        h_backward = h_n[-1] # (B, H)
        sent_vec = torch.cat([h_forward, h_backward], dim=1)  # (B, 2H)

        sent_vec = self.dropout(sent_vec)
        logits = self.classifier(sent_vec)  # (B, 1)
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
        lengths = batch["lengths"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, lengths)
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

    model = BiLSTMClassifier(
        emb_matrix=emb_matrix,
        pad_id=PAD_ID,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        lstm_dropout=LSTM_DROPOUT,
        cls_dropout=CLS_DROPOUT,
        freeze_emb=FREEZE_EMB,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            lengths = batch["lengths"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / max(1, total)
        dev_metrics = evaluate(model, dev_loader, criterion)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} | dev_loss={dev_metrics['loss']:.4f} | dev_acc={dev_metrics['acc']:.4f}"
        )

    # 保存（必要なら）
    save_path = os.path.join(os.path.dirname(TRAIN_PKL_PATH), "out_77_model_bilstm.pt")
    torch.save(model.state_dict(), save_path)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
