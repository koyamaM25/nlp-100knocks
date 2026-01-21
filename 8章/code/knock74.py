# out_74_eval_all_epochs.py
# out_73_fc_epochXX.pt を全て読み込み、devで評価(loss/acc)を一覧表示する

import os
import re
import pickle
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ========= パス設定 =========
EMB_MATRIX_PATH = "/home/koyama/nlp-100knocks/8章/out/out_70_embedding_matrix.npy"
DEV_PKL_PATH    = "/home/koyama/nlp-100knocks/8章/out/out_71_dev.pkl"
OUT_DIR         = "/home/koyama/nlp-100knocks/8章/out"

# fc-only checkpoint のファイル名パターン
CKPT_REGEX = re.compile(r"^out_73_fc_epoch(\d+)\.pt$")

PAD_ID = 0
BATCH_SIZE = 64
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========= Dataset / collate =========
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


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_ids_list = [x["input_ids"] for x in batch]
    labels_list = [x["label"] for x in batch]

    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=PAD_ID
    )  # (B, Lmax)

    labels = torch.stack([t.reshape(()) for t in labels_list], dim=0).float()  # (B,)

    return {"input_ids": input_ids, "labels": labels}


# ========= Model =========
class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, emb_matrix: np.ndarray, pad_id: int = 0, freeze: bool = True):
        super().__init__()
        emb_tensor = torch.tensor(emb_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            emb_tensor,
            freeze=freeze,
            padding_idx=pad_id,
        )
        emb_dim = emb_tensor.shape[1]
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # (B, L, D)

        mask = (input_ids != PAD_ID).unsqueeze(-1).float()  # (B, L, 1)
        x = x * mask

        lengths = mask.sum(dim=1).clamp(min=1.0)            # (B, 1)
        sent_vec = x.sum(dim=1) / lengths                   # (B, D)

        logits = self.fc(sent_vec).squeeze(-1)              # (B,)
        return logits


# ========= Evaluate =========
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids)
        loss = criterion(logits, labels)

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


def list_checkpoints(out_dir: str) -> List[Tuple[int, str]]:
    """OUT_DIRから out_73_fc_epochXX.pt を見つけて epoch順で返す"""
    found = []
    for fn in os.listdir(out_dir):
        m = CKPT_REGEX.match(fn)
        if m:
            epoch = int(m.group(1))
            found.append((epoch, os.path.join(out_dir, fn)))
    found.sort(key=lambda x: x[0])
    return found


def main():
    print(f"device: {DEVICE}")

    # dev data
    with open(DEV_PKL_PATH, "rb") as f:
        dev_data = pickle.load(f)

    dev_loader = DataLoader(
        SSTDataset(dev_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # embedding matrix（固定。毎回同じものを使う）
    emb_matrix = np.load(EMB_MATRIX_PATH)

    # checkpoint一覧
    ckpts = list_checkpoints(OUT_DIR)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched in {OUT_DIR}: out_73_fc_epochXX.pt")

    print(f"found {len(ckpts)} checkpoints in {OUT_DIR}\n")

    results = []
    best = None  # (acc, epoch, path, loss)

    for epoch, path in ckpts:
        # 毎epoch、同一構造のモデルを作って fc を読み込む
        model = MeanEmbeddingClassifier(emb_matrix, pad_id=PAD_ID, freeze=True).to(DEVICE)

        ckpt = torch.load(path, map_location=DEVICE)
        if "fc_state_dict" not in ckpt:
            raise KeyError(f"'fc_state_dict' not found in {path}. keys={list(ckpt.keys())}")

        model.fc.load_state_dict(ckpt["fc_state_dict"])

        metrics = evaluate(model, dev_loader)
        results.append((epoch, metrics["loss"], metrics["acc"], path))

        if best is None or metrics["acc"] > best[0]:
            best = (metrics["acc"], epoch, path, metrics["loss"])

        print(f"[epoch {epoch:02d}] dev_loss={metrics['loss']:.4f} dev_acc={metrics['acc']:.4f}  ({os.path.basename(path)})")

    print("\n=== summary ===")
    # 表っぽく
    print("epoch\tdev_loss\tdev_acc\tcheckpoint")
    for epoch, loss, acc, path in results:
        print(f"{epoch:02d}\t{loss:.4f}\t{acc:.4f}\t{os.path.basename(path)}")

    if best is not None:
        best_acc, best_epoch, best_path, best_loss = best
        print("\n=== best ===")
        print(f"best_epoch : {best_epoch:02d}")
        print(f"best_acc   : {best_acc:.4f}")
        print(f"best_loss  : {best_loss:.4f}")
        print(f"ckpt       : {best_path}")


if __name__ == "__main__":
    main()
