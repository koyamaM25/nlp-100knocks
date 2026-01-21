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

FREEZE_EMB = False

# TextCNN設定
KERNEL_SIZES = [3, 4, 5]
NUM_FILTERS = 100          # 各kernelの出力チャネル数
DROPOUT_P = 0.5            # Kim(2014)でよく使われる
USE_PADDING_IN_CONV = True # 短文対策で same-ish にするなら True

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
# collate_fn（可変長PAD）
# =========================
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_ids_list = [x["input_ids"] for x in batch]
    labels_list = [x["label"] for x in batch]

    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=PAD_ID
    )  # (B, Lmax)

    attention_mask = (input_ids != PAD_ID).long()  # 使わなくても良いが確認用に返す
    labels = torch.stack(labels_list, dim=0).float()  # (B, 1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# =========================
# モデル：Embedding + TextCNN + Linear
# =========================
class TextCNNClassifier(nn.Module):
    def __init__(
        self,
        emb_matrix: np.ndarray,
        pad_id: int = 0,
        kernel_sizes: List[int] = None,
        num_filters: int = 100,
        dropout_p: float = 0.5,
        freeze_emb: bool = False,
        use_padding_in_conv: bool = True,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]

        emb_tensor = torch.from_numpy(emb_matrix)  # float32 (V, D)
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=emb_tensor,
            freeze=freeze_emb,
            padding_idx=pad_id
        )
        emb_dim = emb_tensor.size(1)

        # Conv1d は (B, channels, length) を期待するので channels=emb_dim
        # padding を入れると短文でも畳み込みが可能になりやすい
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = (k // 2) if use_padding_in_conv else 0
            self.convs.append(nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=k, padding=padding))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

        # 各kernelの pooled (B, num_filters) を concat → (B, num_filters * len(kernel_sizes))
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, L)
        return   : logits (B, 1)
        """
        x = self.embedding(input_ids)   # (B, L, D)
        x = x.transpose(1, 2)           # (B, D, L)  ← Conv1d用

        pooled_list = []
        for conv in self.convs:
            h = conv(x)                 # (B, C, L')  C=num_filters
            h = self.relu(h)
            # Global Max Pooling（時間方向）
            h = torch.max(h, dim=2).values  # (B, C)
            pooled_list.append(h)

        feat = torch.cat(pooled_list, dim=1)  # (B, C * num_kernels)
        feat = self.dropout(feat)
        logits = self.classifier(feat)        # (B, 1)
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

    model = TextCNNClassifier(
        emb_matrix=emb_matrix,
        pad_id=PAD_ID,
        kernel_sizes=KERNEL_SIZES,
        num_filters=NUM_FILTERS,
        dropout_p=DROPOUT_P,
        freeze_emb=FREEZE_EMB,
        use_padding_in_conv=USE_PADDING_IN_CONV,
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
    save_path = os.path.join(os.path.dirname(TRAIN_PKL_PATH), "out_78_model_textcnn.pt")
    torch.save(model.state_dict(), save_path)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
