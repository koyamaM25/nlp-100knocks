import os
import pandas as pd
from gensim.models import KeyedVectors
import numpy as np
from sklearn.cluster import KMeans

vec_path = "/home/koyama/nlp-100knocks/6章/code/GoogleNews-vectors-negative300.bin.gz"
capital_path = "/home/koyama/nlp-100knocks/6章/code/questions-words.txt"
output_path = "/home/koyama/nlp-100knocks/6章/out"

os.makedirs(output_path, exist_ok=True)
output_file = os.path.join(output_path, "out_57.txt")

print("Loading model...")
model = KeyedVectors.load_word2vec_format(vec_path, binary=True)

#  国名の抽出
countries = set()

with open(capital_path, 'r', encoding='utf-8') as f:
    is_target_section = False
    
    for line in f:
        line = line.strip()
        if not line: continue
        
        if line.startswith(':'):
            if 'capital-common-countries' in line:
                is_target_section = True
            else:
                is_target_section = False
            continue
        
        # 対象セクション内の行なら国名を取得
        if is_target_section:
            parts = line.split()
            # 2列目(parts[1]) と 4列目(parts[3]) が国名
            countries.add(parts[1])
            countries.add(parts[3])

countries = list(countries) 
print(f"Extracted {len(countries)} countries.")

#  ベクトルの取得
country_vecs = []
valid_countries = []

for country in countries:
    if country in model:
        country_vecs.append(model[country])
        valid_countries.append(country)

#  k-meansクラスタリング
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(country_vecs)

#  結果の表示と保存
labels = kmeans.labels_

# ファイルを開いた状態でループを回して、全クラスタの結果を書き込みます
with open(output_file, 'w', encoding='utf-8') as f:
    for i in range(5):
        # ラベルが i である国のリストを作成
        cluster_countries = [valid_countries[j] for j in range(len(valid_countries)) if labels[j] == i]
        
        print(f"\n=== Cluster {i} ===")
        print(", ".join(cluster_countries))
        
        f.write(f"\n=== Cluster {i} ===\n")
        f.write(", ".join(cluster_countries) + "\n")

print(f"\nSaved results to {output_file}")