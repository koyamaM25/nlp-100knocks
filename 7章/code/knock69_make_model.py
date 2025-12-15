from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import os
import json
import joblib

vectorizer_file = "/home/koyama/nlp-100knocks/7章/out/out_62_vectorize.joblib"
input_file_t = "/home/koyama/nlp-100knocks/7章/out/out_61_train.json"
input_file_d = "/home/koyama/nlp-100knocks/7章/out/out_61_dev.json"
output_path = "/home/koyama/nlp-100knocks/7章/out"
os.makedirs(output_path, exist_ok=True)

with open(input_file_t, "r") as f_t,\
     open(input_file_d, "r") as f_d:
    train_data = json.load(f_t)
    dev_data = json.load(f_d)

#特徴量(X)とラベル(y)のリストを作成
X_train_dicts = [item['feature'] for item in train_data]
y_train = [item['label'] for item in train_data]

X_dev_dicts = [item['feature'] for item in dev_data]
y_dev = [item['label'] for item in dev_data]

#ベクトル化
vectorizer = joblib.load(vectorizer_file)
print("Vectorizing...")
X_train = vectorizer.transform(X_train_dicts)
X_dev = vectorizer.transform(X_dev_dicts)

#ロジスティック回帰モデルの学習
def make_Logisic_joblib(X_train, y_train, c):
    model = LogisticRegression(C=c, solver='liblinear', random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    c_str = str(c)
    C_list = []
    for s in c_str:
        if s.isnumeric():
            C_list.append(s)
    C_str = "".join(C_list)
    model_name = f"out_69_model_c{C_str}.joblib"
    save_path = os.path.join(output_path, model_name)
    joblib.dump(model, save_path)

print("now Training loop...")
c_values = [0.01, 0.1, 1, 10, 100]
for c in c_values:
    make_Logisic_joblib(X_train, y_train, c)
