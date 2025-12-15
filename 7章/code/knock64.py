import os
import json
import joblib

input_file = "/home/koyama/nlp-100knocks/7章/out/out_61_dev.json"
model_file = "/home/koyama/nlp-100knocks/7章/out/out_62_model.joblib"
vectorizer_file = "/home/koyama/nlp-100knocks/7章/out/out_62_vectorize.joblib"
output_path = "/home/koyama/nlp-100knocks/7章/out"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "out_64.txt")

model = joblib.load(model_file)
vectorizer = joblib.load(vectorizer_file)

with open(input_file, "r") as f:
    dev_data = json.load(f)

text = dev_data[0]['feature']
gold_label = dev_data[0]['label']

#ベクトル化
print(f"vectorize")
X = vectorizer.transform([text])

#予測_条件付確率
print(f"now Predicting ...")
proba = model.predict_proba(X)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"proba\tgold\n")
    f.write(f"{proba}\t{gold_label}\n")
