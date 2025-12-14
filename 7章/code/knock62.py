from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import os
import json
import joblib

input_file = "/home/koyama/nlp-100knocks/7章/out/out_61_train.json"
output_path = "/home/koyama/nlp-100knocks/7章/out"
os.makedirs(output_path, exist_ok=True)

model_file = os.path.join(output_path, "out_62_model.joblib")
vectorizer_file = os.path.join(output_path, "out_62_vectorize.joblib")

with open(input_file, "r") as f:
    train_data = json.load(f)

#特徴量(X)とラベル(y)のリストを作成
X_train_dicts = [item['feature'] for item in train_data]
y_train = [item['label'] for item in train_data]

#DictVectorizerで辞書を行列に変換
#辞書にある単語を列（カラム）に割り当て、出現回数を値に入れた行列を作る
print(f"vectorize")
vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(X_train_dicts)

#ロジスティック回帰モデルの学習
print(f"now Learning ...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

joblib.dump(model, model_file)
joblib.dump(vectorizer, vectorizer_file)