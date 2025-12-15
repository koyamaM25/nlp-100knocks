import os
import json
import joblib
from sklearn.metrics import confusion_matrix

input_file = "/home/koyama/nlp-100knocks/7章/out/out_61_dev.json"
model_file = "/home/koyama/nlp-100knocks/7章/out/out_62_model.joblib"
vectorizer_file = "/home/koyama/nlp-100knocks/7章/out/out_62_vectorize.joblib"
output_path = "/home/koyama/nlp-100knocks/7章/out"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "out_67.txt")

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

#混同行列
cm = confusion_matrix(labels, pred_labels)

#正解率，適合率，再現率，F1スコア
#cm は [[TN, FP], [FN, TP]] の順
tn, fp, fn, tp = cm.flatten()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
F1 = (2 * precision * recall) / (precision + recall)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"正解率\t適合率\t再現率\tF1スコア\n")
    f.write(f"{accuracy}\t{precision}\t{recall}\t{F1}\n")
