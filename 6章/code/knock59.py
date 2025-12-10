from gensim.models import KeyedVectors
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ===== パス設定 =====
vec_path = "/home/koyama/nlp-100knocks/6章/code/GoogleNews-vectors-negative300.bin.gz"
questions_path = "/home/koyama/nlp-100knocks/6章/code/questions-words.txt"
output_dir = "/home/koyama/nlp-100knocks/6章/out"

os.makedirs(output_dir, exist_ok=True)
fig_path = os.path.join(output_dir, "out_59.png")

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

        # セクション行
        if line.startswith(":"):
            # : capital-common-countries だけを使う
            if line.lower().startswith(": capital-common-countries"):
                use_block = True
            else:
                use_block = False
            continue

        # capital-common-countries 以外は無視
        if not use_block:
            continue

        # 形式: capital1 country1 capital2 country2
        parts = line.split()
        if len(parts) != 4:
            continue

        _, country1, _, country2 = parts
        countries.extend([country1, country2])

# 重複削除＋語彙外(OOV)除去
unique_countries = []
seen = set()
for c in countries:
    if c in model.key_to_index and c not in seen:
        seen.add(c)
        unique_countries.append(c)

print(f"国の数（モデル語彙に存在）: {len(unique_countries)}")

if len(unique_countries) < 3:
    raise ValueError("t-SNE を適用するには国の数が少なすぎます。")

# ===== 3. ベクトル行列の作成 =====
X = np.array([model[c] for c in unique_countries])

# ===== 4. t-SNE による 2 次元埋め込み =====
# perplexity はサンプル数未満である必要があるので自動調整
n_samples = len(unique_countries)
perplexity = min(10, n_samples - 1)

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    learning_rate="auto",
    init="pca",
    random_state=0,
)

print(f"Running t-SNE (perplexity={perplexity}, n_samples={n_samples}) ...")
X_embedded = tsne.fit_transform(X)

# ===== 5. 可視化 =====
plt.figure(figsize=(12, 8))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1])

for (x, y), label in zip(X_embedded, unique_countries):
    plt.text(x, y, label, fontsize=8)

plt.title("t-SNE visualization of country word vectors")
plt.tight_layout()
plt.savefig(fig_path)
plt.close()

print(f"t-SNE 図を {fig_path} に保存しました。")
