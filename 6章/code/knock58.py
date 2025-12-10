from gensim.models import KeyedVectors
import os
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# ===== パス設定 =====
vec_path = "/home/koyama/nlp-100knocks/6章/code/GoogleNews-vectors-negative300.bin.gz"
questions_path = "/home/koyama/nlp-100knocks/6章/code/questions-words.txt"
output_dir = "/home/koyama/nlp-100knocks/6章/out"

os.makedirs(output_dir, exist_ok=True)
fig_path = os.path.join(output_dir, "out_58.png")

# ===== 1. 単語ベクトルモデルの読み込み =====
print("Loading word2vec model ...")
model = KeyedVectors.load_word2vec_format(vec_path, binary=True)
print("Model loaded.")

# ===== 2. capital-common-countries セクションから国名を抽出 =====
countries = []
use_block = False

with open(questions_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # セクションの切り替え
        if line.startswith(":"):
            # capital-common-countries セクションかどうか
            if line.lower().startswith(": capital-common-countries"):
                use_block = True
            else:
                # 別カテゴリに移ったらオフ
                use_block = False
            continue

        if not use_block:
            continue

        # 行は「capital1 country1 capital2 country2」の4語
        parts = line.split()
        if len(parts) != 4:
            continue

        _, country1, _, country2 = parts[0], parts[1], parts[2], parts[3]
        countries.extend([country1, country2])

# 重複削除＋語彙外(OOV)除去（順序は保持）
unique_countries = []
seen = set()
for c in countries:
    if c in model.key_to_index and c not in seen:
        seen.add(c)
        unique_countries.append(c)

print(f"国の数（モデル語彙に存在）: {len(unique_countries)}")

# ===== 3. ベクトル行列の作成 =====
X = np.array([model[c] for c in unique_countries])

# ===== 4. Ward 法による階層型クラスタリング =====
# linkage の method='ward' が Ward 法
Z = linkage(X, method="ward")

# ===== 5. デンドログラムの描画・保存 =====
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=unique_countries, leaf_rotation=90)
plt.title("Hierarchical Clustering of Countries (Ward method)")
plt.tight_layout()
plt.savefig(fig_path)
plt.close()

print(f"デンドログラムを {fig_path} に保存しました。")
