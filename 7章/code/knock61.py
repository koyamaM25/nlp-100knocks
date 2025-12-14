import zipfile
import pandas as pd
from collections import Counter
import os
import json

input_file = "/home/koyama/nlp-100knocks/7章/code/SST-2.zip"
output_path = "/home/koyama/nlp-100knocks/7章/out"
os.makedirs(output_path, exist_ok=True)

output_file_t = os.path.join(output_path, "out_61_train.json") 
output_file_d = os.path.join(output_path, "out_61_dev.json")

with zipfile.ZipFile(input_file, "r") as z:
    with z.open('SST-2/train.tsv') as f:
        df_train = pd.read_csv(f, sep='\t')
    with z.open('SST-2/dev.tsv') as f:
        df_dev = pd.read_csv(f, sep='\t')

#単語頻度
def create_features(text):
    words = text.split()
    return dict(Counter(words))

#データ作成
def create_data(df, data):
    for index, row in df.iterrows():
        item = {
            'text': row['sentence'],
            'label': row['label'],
            'feature': create_features(row['sentence'])
        }
        data.append(item)

train_data = []
dev_data = []

create_data(df_train, train_data)
create_data(df_dev, dev_data)

print("学習データの最初の事例:")
print(train_data[0])

with open(output_file_t, 'w', encoding='utf-8') as f_t, \
     open(output_file_d, 'w', encoding='utf-8') as f_d: 

    json.dump(train_data, f_t, ensure_ascii=False, indent=2)
    json.dump(dev_data, f_d, ensure_ascii=False, indent=2)