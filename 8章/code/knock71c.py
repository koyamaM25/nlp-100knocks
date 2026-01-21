import os
import re
import pandas as pd
import zipfile
import pickle
from typing import List, Dict, Any, Tuple

import torch

WORD2ID_PATH = "/home/koyama/nlp-100knocks/8章/out/out_70_word2id.pkl"
SST2_ZIP_PATH = "/home/koyama/nlp-100knocks/7章/code/SST-2.zip"

OUT_DIR = "/home/koyama/nlp-100knocks/8章/out"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TRAIN = os.path.join(OUT_DIR, "out_71_train.pkl")
OUT_DEV = os.path.join(OUT_DIR, "out_71_dev.pkl")

# トークナイザ
# GoogleNews word2vec は大文字小文字を区別するため lower() はしない
# 以下の正規表現で以下を抽出する：
# ・英単語（don't などのアポストロフィ付きも可）
# ・数字
# ・記号1文字
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]")

def tokenize(text: str) -> List[str]:
    """文をトークン列に分割する"""
    return TOKEN_RE.findall(text)

# SST-2 を zip から読み込む
def load_sst2_from_zip(zip_path: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    SST-2 の train.tsv / dev.tsv を zip から読み込み，
    (文, ラベル) のリスト（train, dev）を返す
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("SST-2/train.tsv") as f:
            df_train = pd.read_csv(f, sep="\t")
        with z.open("SST-2/dev.tsv") as f:
            df_dev = pd.read_csv(f, sep="\t")

    # (sentence, label) へ変換
    train_examples = list(zip(df_train["sentence"].tolist(), df_train["label"].astype(int).tolist()))
    dev_examples = list(zip(df_dev["sentence"].tolist(), df_dev["label"].astype(int).tolist()))

    return train_examples, dev_examples

# token id 列への変換
def convert_examples(
    examples: List[Tuple[str, int]],
    word2id: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    (文, ラベル) を以下の辞書形式へ変換する：
    {
        "text": 元文,
        "label": tensor([0.]) or tensor([1.]),
        "input_ids": token id の tensor
    }

    語彙外のみで空になった文は除外する

    戻り値：
      dataset, 除外数, 全体数
    """
    dataset = []
    dropped = 0
    total = 0

    for text, label_int in examples:
        total += 1

        # 文をトークン化
        tokens = tokenize(text)

        # word2id に存在する単語のみ token id に変換（語彙外は無視）
        ids = [word2id[t] for t in tokens if t in word2id]

        # 全単語が語彙外なら除外
        if len(ids) == 0:
            dropped += 1
            continue

        item = {
            "text": text,
            "label": torch.tensor([float(label_int)], dtype=torch.float32),
            "input_ids": torch.tensor(ids, dtype=torch.long),
        }
        dataset.append(item)

    return dataset, dropped, total

def main():
    # 問題70で作成した word2id を読み込む
    with open(WORD2ID_PATH, "rb") as f:
        word2id = pickle.load(f)

    # SST-2 を zip から読み込む（train/dev）
    train_examples, dev_examples = load_sst2_from_zip(SST2_ZIP_PATH)

    # token id 列へ変換し，空文を除外
    train_data, train_dropped, train_total = convert_examples(train_examples, word2id)
    dev_data, dev_dropped, dev_total = convert_examples(dev_examples, word2id)

    # pickle で保存
    with open(OUT_TRAIN, "wb") as f:
        pickle.dump(train_data, f)

    with open(OUT_DEV, "wb") as f:
        pickle.dump(dev_data, f)

    # 結果表示
    print("=== 問題71 完了 ===")
    print(f"語彙サイズ        : {len(word2id):,}")
    print(f"train 全体数      : {train_total:,}")
    print(f"train 使用数      : {len(train_data):,}")
    print(f"train 除外数      : {train_dropped:,} ({train_dropped/train_total*100:.2f}%)")
    print(f"dev 全体数        : {dev_total:,}")
    print(f"dev 使用数        : {len(dev_data):,}")
    print(f"dev 除外数        : {dev_dropped:,} ({dev_dropped/dev_total*100:.2f}%)")
    print(f"保存先(train)     : {OUT_TRAIN}")
    print(f"保存先(dev)       : {OUT_DEV}")
    print(train_data[:3])

if __name__ == "__main__":
    main()
