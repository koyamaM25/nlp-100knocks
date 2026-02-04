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
# Dataloaderにするための前処理
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

train_dataset = CreateDataset(df_train['input_ids'].tolist(), df_train['label'].tolist())
dev_dataset = CreateDataset(df_dev['input_ids'].tolist(), df_dev['label'].tolist())

print(train_dataset[:1])

# DataLoaderを作成
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)



# ↓以下ファインチューニング部分
from transformers import AutoModelForSequenceClassification
import torch.nn as nn
import torch.optim as optim
import numpy as np

MODEL_DIR =os.path.join(OUTPUT_PATH, 'model_87')
os.makedirs(MODEL_DIR, exist_ok=True)

# num_labels=2 は「ポジティブ・ネガティブ」の2値分類だから
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# 学習の設定
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# 学習と検証
def train_model(model, train_loader, val_loader, epochs=1):
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        # 訓練
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # データをGPUへ
            b_input_ids = batch['input_ids'].to(device)
            b_labels = batch['labels'].to(device)
            
            # 勾配（計算の履歴）をリセット
            optimizer.zero_grad()
            
            # 順伝播: 予測させる
            # labelsを渡すと、内部で自動的にLossも計算して返してくれます
            outputs = model(b_input_ids, labels=b_labels)
            loss = outputs.loss
            
            # 逆伝播: 誤差から修正量を計算
            loss.backward()
            
            # パラメータ更新
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")
        
        # 検証
        model.eval() 
        preds_list = []
        labels_list = []
        
        # 評価時は勾配計算不要
        with torch.no_grad():
            for batch in val_loader:
                b_input_ids = batch['input_ids'].to(device)
                b_labels = batch['labels'].to(device)
                
                outputs = model(b_input_ids)
                logits = outputs.logits
                
                # スコアが一番高いクラスを選ぶ（予測）
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = b_labels.cpu().numpy()
                
                preds_list.extend(preds)
                labels_list.extend(labels)
        
        # 正解率計算
        accuracy = np.sum(np.array(preds_list) == np.array(labels_list)) / len(labels_list)
        print(f"Validation Accuracy: {accuracy:.4f}")

# 実行
train_model(model, train_dataloader, dev_dataloader, epochs=3)

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
