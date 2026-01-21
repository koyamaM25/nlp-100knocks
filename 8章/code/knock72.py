import os
import pickle
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

EMB_MATRIX_PATH = "/home/koyama/nlp-100knocks/8章/out/out_70_embedding_matrix.npy"
TRAIN_PATH = "/home/koyama/nlp-100knocks/8章/out/out_71_train.pkl"
DEV_PATH   = "/home/koyama/nlp-100knocks/8章/out/out_71_dev.pkl"

OUT_DIR = "/home/koyama/nlp-100knocks/8章/out"
os.makedirs(OUT_DIR, exist_ok=True)


class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, emb_matrix: np.ndarray, freeze: bool = True):
        super().__init__()
        emb_tensor = torch.tensor(emb_matrix, dtype=torch.float32)

        self.embedding = nn.Embedding.from_pretrained(
            emb_tensor,
            freeze=freeze,
            padding_idx=0,
        )
        emb_dim = emb_tensor.shape[1]
        self.fc = nn.Linear(emb_dim, 1)  # 2値なので1ユニット（BCEWithLogitsLoss想定）

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, L) の LongTensor（padding込み）
        returns: logits (B,) の FloatTensor
        """
        # (B, L, D)
        x = self.embedding(input_ids)

        # PAD(=0) を除外して平均
        mask = (input_ids != 0).unsqueeze(-1).float()  # (B, L, 1)
        x = x * mask                                   # PAD位置を0に

        # 長さ（PAD以外のトークン数）
        lengths = mask.sum(dim=1)                      # (B, 1)
        lengths = lengths.clamp(min=1.0)               # 念のため0割防止

        sent_vec = x.sum(dim=1) / lengths              # (B, D)
        logits = self.fc(sent_vec).squeeze(-1)         # (B,)
        return logits


def load_pkl(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def pad_batch(batch_input_ids: List[torch.Tensor], pad_id: int = 0) -> torch.Tensor:
    B = len(batch_input_ids)
    Lmax = max(x.size(0) for x in batch_input_ids)
    out = torch.full((B, Lmax), pad_id, dtype=torch.long)
    for i, x in enumerate(batch_input_ids):
        out[i, : x.size(0)] = x
    return out


def main():
    print("Loading embedding matrix ...")
    emb_matrix = np.load(EMB_MATRIX_PATH)  # (V, D)

    print("Loading datasets ...")
    train_data = load_pkl(TRAIN_PATH)
    dev_data = load_pkl(DEV_PATH)

    # モデル作成（問72）
    model = MeanEmbeddingClassifier(emb_matrix, freeze=True)
    model.eval()

    # forward動作確認（ミニバッチっぽく3件）
    sample = train_data[:3]
    batch_ids = [item["input_ids"] for item in sample]  # 1D tensorたち
    batch = pad_batch(batch_ids, pad_id=0)              # (B, L)

    with torch.no_grad():
        logits = model(batch)

    print("=== 問72 forward check ===")
    print(f"batch shape      : {batch.shape}")
    print(f"logits shape     : {logits.shape}")
    print(f"logits           : {logits}")
    print(f"labels (raw)     : {[item['label'].item() for item in sample]}")


if __name__ == "__main__":
    main()
