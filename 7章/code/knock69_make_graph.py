import os
import json
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

input_file_t = "/home/koyama/nlp-100knocks/7章/out/out_61_train.json"
input_file_d = "/home/koyama/nlp-100knocks/7章/out/out_61_dev.json"
vectorizer_file = "/home/koyama/nlp-100knocks/7章/out/out_62_vectorize.joblib"
output_path = "/home/koyama/nlp-100knocks/7章/out"
os.makedirs(output_path, exist_ok=True)

output_png = os.path.join(output_path, "out_69_graph.png")
output_txt = os.path.join(output_path, "out_69.txt")

vectorizer = joblib.load(vectorizer_file)

with open(input_file_t, "r") as f_t,\
     open(input_file_d, "r") as f_d:
    train_data = json.load(f_t)
    dev_data = json.load(f_d)

train_texts = [d['feature'] for d in train_data]
train_labels = [d['label'] for d in train_data]
dev_texts = [d['feature'] for d in dev_data]
dev_labels = [d['label'] for d in dev_data]

#ベクトル化
print(f"Vectorizing...")
X_train = vectorizer.transform(train_texts)
X_dev = vectorizer.transform(dev_texts)

def get_accracy(model_file, X, labels):
    #予測
    model_path = os.path.join(output_path, model_name)
    model = joblib.load(model_path)
    pred_labels = model.predict(X)
    #混同行列
    cm = confusion_matrix(labels, pred_labels)

    #正解率
    #cm は [[TN, FP], [FN, TP]] の順
    tn, fp, fn, tp = cm.flatten()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy

#推測ループ
c_values = [0.01, 0.1, 1, 10, 100]
print(f"now Predicting ...")
accuracy_train = []
accuracy_dev = []
for c in c_values:
    c_str = str(c)
    C_list = []
    for s in c_str:
        if s.isnumeric():
            C_list.append(s)
    C_str = "".join(C_list)
    model_name = f"out_69_model_c{C_str}.joblib"
    accuracy_train.append(get_accracy(model_name, X_train, train_labels))
    accuracy_dev.append(get_accracy(model_name, X_dev, dev_labels))

#描画
plt.figure(figsize=(8, 6))
plt.plot(c_values, accuracy_train, label='Train Accuracy', marker='o')
plt.plot(c_values, accuracy_dev, label='Dev Accuracy', marker='x')

plt.xscale('log') # 横軸を対数スケールに
plt.xlabel('C (Regularization parameter)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Regularization Parameter C')
plt.legend()
plt.grid(True)

plt.savefig(output_png)

with open(output_txt, "w", encoding="utf-8") as f:
    f.write(f"accuracy_train\n")
    for accuracy,c in zip(accuracy_train, c_values):
        f.write(f"accuracy {c} = {accuracy}\n")
    
    f.write(f"\n")

    f.write(f"accuracy_dev\n")
    for accuracy,c in zip(accuracy_dev, c_values):
        f.write(f"accuracy {c} = {accuracy}\n")
