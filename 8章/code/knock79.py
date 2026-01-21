import os
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

# 問題79：アーキテクチャ変更（TextCNN）
KERNEL_SIZES = [3, 4, 5]
NUM_FILTERS = 100      # 各カーネルのフィルタ数
DROPOUT_P = 0.5        # TextCNNでよく使う
FINE_TUNE_EMB = True   # True: 問題78の要素も含む（埋め込み更新）/ False: 固定

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
# Dataset（問題71の出力形式を想定）
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
# collate（問題75に準拠：PAD＋長い順に並び替え）
# =========================
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # 長い順にソート（問題75の指示）
    batch = sorted(batch, key=lambda x: x["input_ids"].size(0), reverse=True)

    input_ids_list = [x["input_ids"] for x in batch]
    labels_list = [x["label"] for x in batch]

    # lengths（将来RNNにする時にも使える）
    lengths = torch.tensor([t.size(0) for t in input_ids_list], dtype=torch.long)

    # PADして (B, Lmax)
    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=PAD_ID
    )

    labels = torch.stack(labels_list, dim=0).float()  # (B, 1)
    attention_mask = (input_ids != PAD_ID).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "labels": labels,
    }


# =========================
# TextCNN モデル
# =========================
class TextCNN(nn.Module):
    def __init__(
        self,
        emb_matrix: np.ndarray,
        pad_id: int = 0,
        kernel_sizes: List[int] = None,
        num_filters: int = 100,
        dropout_p: float = 0.5,
        fine_tune_emb: bool = True,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]

        emb_tensor = torch.from_numpy(emb_matrix)  # (V, D) float32

        # fine_tune_emb=True なら埋め込み更新（問題78の発展要素）
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=emb_tensor,
            freeze=not fine_tune_emb,
            padding_idx=pad_id
        )
        emb_dim = emb_tensor.size(1)

        # Conv1d: 入力は (B, D, L)
        # padding=k//2 で短文にも比較的強くする（厳密な"same"ではないが実用上OK）
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # (B, L, D)
        x = self.embedding(input_ids)

        # Conv1d用に (B, D, L)
        x = x.transpose(1, 2)

        pooled = []
        for conv in self.convs:
            h = conv(x)                 # (B, C, L')
            h = self.relu(h)
            h = torch.max(h, dim=2).values  # Global max pooling -> (B, C)
            pooled.append(h)

        feat = torch.cat(pooled, dim=1)     # (B, C * |K|)
        feat = self.dropout(feat)
        logits = self.classifier(feat)      # (B, 1)
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
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids)
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

    # 埋め込み行列（問題70）
    emb_matrix = np.load(EMB_MATRIX_PATH)  # (V, D)
    print("embedding matrix:", emb_matrix.shape)

    # データ（問題71）
    with open(TRAIN_PKL_PATH, "rb") as f:
        train_data = pickle.load(f)
    with open(DEV_PKL_PATH, "rb") as f:
        dev_data = pickle.load(f)

    train_loader = DataLoader(
        SSTDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        SSTDataset(dev_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    model = TextCNN(
        emb_matrix=emb_matrix,
        pad_id=PAD_ID,
        kernel_sizes=KERNEL_SIZES,
        num_filters=NUM_FILTERS,
        dropout_p=DROPOUT_P,
        fine_tune_emb=FINE_TUNE_EMB,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids)
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
    save_path = os.path.join(os.path.dirname(TRAIN_PKL_PATH), "out_79_model_textcnn.pt")
    torch.save(model.state_dict(), save_path)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
