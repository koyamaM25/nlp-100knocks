import os
import joblib

model_file = "/home/koyama/nlp-100knocks/7章/out/out_62_model.joblib"
vectorizer_file = "/home/koyama/nlp-100knocks/7章/out/out_62_vectorize.joblib"
output_path = "/home/koyama/nlp-100knocks/7章/out"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "out_68.txt")

model = joblib.load(model_file)
vectorizer = joblib.load(vectorizer_file)

#単語リストを取得
feature_names = vectorizer.get_feature_names_out()

#単語と重みをペアにしたリスト作成
features = []
for weight, name in zip(model.coef_[0], feature_names):
    features.append((name, weight))

#重みによって降順ソート
features.sort(key=lambda x : x[1])

#上位20件と下位20件の定義
top20 = sorted(features[-20:], key=lambda x : x[1], reverse=True)
bottom20 = features [:20]

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"top20\n")
    for word, weight in top20:
        f.write(f"{word}\t{weight}\n")
    f.write(f"\n")
    f.write(f"bottom20\n")
    for word, weight in bottom20:
        f.write(f"{word}\t{weight}\n")

    
