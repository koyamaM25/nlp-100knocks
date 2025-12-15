import os
import joblib
from collections import Counter

model_file = "/home/koyama/nlp-100knocks/7章/out/out_62_model.joblib"
vectorizer_file = "/home/koyama/nlp-100knocks/7章/out/out_62_vectorize.joblib"
output_path = "/home/koyama/nlp-100knocks/7章/out"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "out_65.txt")

model = joblib.load(model_file)
vectorizer = joblib.load(vectorizer_file)

text = "the worst movie I 've ever seen"

def create_features(text):
    words = text.split()
    return dict(Counter(words))

text = create_features(text)

#ベクトル化
print(f"vectorize")
X = vectorizer.transform([text])

#予測
print(f"now Predicting ...")
pred = model.predict(X)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"pred\n")
    f.write(f"{pred[0]}\n")
