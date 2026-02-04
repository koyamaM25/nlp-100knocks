from transformers import AutoTokenizer
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader

OUTPUT_PATH = "/home/koyama/nlp-100knocks/9章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)

df_train = pd.read_pickle(os.path.join(OUTPUT_PATH, 'out_85_train.pkl'))
df_dev = pd.read_pickle(os.path.join(OUTPUT_PATH, 'out_85_dev.pkl'))

# 最大長を調べる
def search_maxlength(tokens_series):
    max_length = 0
    for tokens in tokens_series:
        if len(tokens) > max_length:
            max_length = len(tokens)
    return max_length

max_len_train = search_maxlength(df_train['tokens'])
max_len_dev = search_maxlength(df_dev['tokens'])

# trainとdevの中で一番長いものに合わせる（+2は [CLS], [SEP] の分）
final_max_length = max(max_len_train, max_len_dev) + 2
print(f"Max length defined as: {final_max_length}")

# トークナイザ関数
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def padding_tokenizer(text, max_len):
    return tokenizer(
        text, 
        truncation=True, 
        padding="max_length",     
        max_length=max_len,       
        return_tensors="pt"
    )

df_train['input_ids'] = df_train['sentence'].apply(lambda x: padding_tokenizer(x, final_max_length)['input_ids'][0])
df_dev['input_ids'] = df_dev['sentence'].apply(lambda x: padding_tokenizer(x, final_max_length)['input_ids'][0])

print(df_train[['sentence', 'input_ids']].head(1))

# Datasetクラスの定義
class CreateDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return {
            'input_ids': self.X[index],
            'labels': torch.tensor(self.y[index], dtype=torch.long)
        }

# ラベルも取得
train_dataset = CreateDataset(df_train['input_ids'].tolist(), df_train['label'].tolist())
dev_dataset = CreateDataset(df_dev['input_ids'].tolist(), df_dev['label'].tolist())

print(train_dataset[:1])
# DataLoaderを作成
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

# 動作確認
batch = next(iter(train_dataloader))
print("\n--- Minibatch Check ---")
print(f"Input IDs shape: {batch['input_ids'].shape}") 
print(f"Labels shape: {batch['labels'].shape}")