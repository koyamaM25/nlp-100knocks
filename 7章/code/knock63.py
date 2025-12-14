import os
import json
import joblib

input_file = "/home/koyama/nlp-100knocks/7章/out/out_61_dev.json"
model_file = "/home/koyama/nlp-100knocks/7章/out/out_62_model.joblib"
vectorizer_file = "/home/koyama/nlp-100knocks/7章/out/out_62_vectorize.joblib"
output_path = "/home/koyama/nlp-100knocks/7章/out"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "out_63.txt")

model = joblib.load(model_file)
vectorizer = joblib.load(vectorizer_file)

with open(input_file, "r") as f:
    dev_data = json.load(f)

texts = [d['feature'] for d in dev_data]
labels = [d['label'] for d in dev_data]

#ベクトル化
print(f"vectorize")
X_dev = vectorizer.transform(texts)

#予測
print(f"now Predicting ...")
pred_labels = model.predict(X_dev)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"pred\tgold\n")
    for pred, gold in zip(pred_labels, labels):
        f.write(f"{pred}\t{gold}\n")
