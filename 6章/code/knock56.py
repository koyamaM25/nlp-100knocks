import os
import zipfile

import pandas as pd
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
import numpy as np

vec_path = "/home/koyama/nlp-100knocks/6章/code/GoogleNews-vectors-negative300.bin.gz"
input_zip = "/home/koyama/nlp-100knocks/6章/code/wordsim353.zip"
output_path = "/home/koyama/nlp-100knocks/6章/out"
os.makedirs(output_path, exist_ok=True)
output_file = os.path.join(output_path, "out_56.txt")

# ===== 1. WordSim353 データの読み込み =====
csv_name_in_zip = "combined.csv"  # zip 内のファイル名に合わせる

with zipfile.ZipFile(input_zip, "r") as zf:
    with zf.open(csv_name_in_zip) as f:
        df = pd.read_csv(f)

print("Columns in combined.csv:", df.columns.tolist())

# ---- 列名のロバストな取得 ----
cols_lower = {c.lower(): c for c in df.columns}

# Word1 / Word 1 / w1 などに対応
def pick_word_col(which: int) -> str:
    cand_patterns = [
        f"word{which}",
        f"word {which}",
        f"w{which}",
    ]
    for key_lower, orig in cols_lower.items():
        for pat in cand_patterns:
            if pat in key_lower:
                return orig
    raise ValueError(f"Word{which} に相当する列が見つかりません: {df.columns.tolist()}")

w1_col = pick_word_col(1)
w2_col = pick_word_col(2)

# 類似度スコア列（Human / Human (mean) / score / similarity など）を探索
score_col = None
for key_lower, orig in cols_lower.items():
    if "human" in key_lower or "score" in key_lower or "similarity" in key_lower:
        score_col = orig
        break

if score_col is None:
    raise ValueError(f"類似度スコア列が特定できません: {df.columns.tolist()}")

print(f"Using columns: w1={w1_col}, w2={w2_col}, score={score_col}")

# ===== 2. 単語ベクトルモデルの読み込み =====
print("Loading word2vec model ...")
model = KeyedVectors.load_word2vec_format(vec_path, binary=True)
print("Model loaded.")

# ===== 3. モデルによる類似度計算 =====
human_scores = []
model_scores = []
oov_pairs = 0

for _, row in df.iterrows():
    w1 = str(row[w1_col])
    w2 = str(row[w2_col])
    human = float(row[score_col])

    # OOV（どちらかが語彙外）ならスキップ
    if (w1 not in model.key_to_index) or (w2 not in model.key_to_index):
        oov_pairs += 1
        continue

    sim = float(model.similarity(w1, w2))
    human_scores.append(human)
    model_scores.append(sim)

human_scores = np.array(human_scores)
model_scores = np.array(model_scores)

# ===== 4. スピアマン相関係数の計算 =====
if len(human_scores) == 0:
    raise RuntimeError("有効なペアが 0 件です（すべて OOV？）")

rho, p_value = spearmanr(human_scores, model_scores)

print(f"Number of total pairs      : {len(df)}")
print(f"Number of OOV pairs        : {oov_pairs}")
print(f"Number of used pairs       : {len(human_scores)}")
print(f"Spearman correlation (rho) : {rho:.4f}")
print(f"p-value                    : {p_value:.4e}")

# ===== 5. ファイル出力 =====
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Total pairs: {len(df)}\n")
    f.write(f"OOV pairs: {oov_pairs}\n")
    f.write(f"Used pairs: {len(human_scores)}\n")
    f.write(f"Spearman correlation (rho): {rho:.6f}\n")
    f.write(f"p-value: {p_value:.6e}\n")
