from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

OUTPUT_PATH = "/home/koyama/nlp-100knocks/9章/out"
MODEL_SAVE_PATH = os.path.join(OUTPUT_PATH, 'model_89_roberta')
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

df_train = pd.read_pickle(os.path.join(OUTPUT_PATH, 'out_85_train.pkl'))
df_dev = pd.read_pickle(os.path.join(OUTPUT_PATH, 'out_85_dev.pkl'))

# --- 変更点1: モデル名を RoBERTa に変更 ---
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 最大長の計算などは同じ
def search_maxlength(tokens_series):
    max_length = 0
    for tokens in tokens_series:
        if len(tokens) > max_length:
            max_length = len(tokens)
    return max_length

final_max_length = max(search_maxlength(df_train['tokens']), search_maxlength(df_dev['tokens'])) + 2

# トークナイザ関数
def get_features(text, max_len):
    enc = tokenizer(
        text, 
        truncation=True, 
        padding="max_length",     
        max_length=max_len,       
        return_tensors="pt"
    )
    return enc['input_ids'][0], enc['attention_mask'][0]

# 前処理の適用
print("Tokenizing data...")
df_train[['input_ids', 'attention_mask']] = df_train['sentence'].apply(
    lambda x: pd.Series(get_features(x, final_max_length))
)
df_dev[['input_ids', 'attention_mask']] = df_dev['sentence'].apply(
    lambda x: pd.Series(get_features(x, final_max_length))
)

# Dataset定義
class CreateDataset(Dataset):
    def __init__(self, X, mask, y):
        self.X = X
        self.mask = mask
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return {
            'input_ids': self.X[index],
            'attention_mask': self.mask[index],
            'labels': torch.tensor(self.y[index], dtype=torch.long)
        }

# Dataset & DataLoader作成
train_dataset = CreateDataset(df_train['input_ids'].tolist(), df_train['attention_mask'].tolist(), df_train['label'].tolist())
dev_dataset = CreateDataset(df_dev['input_ids'].tolist(), df_dev['attention_mask'].tolist(), df_dev['label'].tolist())

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

# --- 変更点2: モデルのロード ---
print(f"Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# 学習ループ
def train_model(model, train_loader, val_loader, epochs=3): # 3エポック推奨
    print(f"Start training with {device}...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        # 訓練
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Training Loss: {total_loss / len(train_loader):.4f}")
        
        # 検証
        model.eval() 
        preds_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                b_input_ids = batch['input_ids'].to(device)
                b_input_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)
                
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                labels = b_labels.cpu().numpy()
                
                preds_list.extend(preds)
                labels_list.extend(labels)
        
        accuracy = np.sum(np.array(preds_list) == np.array(labels_list)) / len(labels_list)
        print(f"Validation Accuracy: {accuracy:.4f}")

# 実行
train_model(model, train_dataloader, dev_dataloader, epochs=3)

# 保存
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Saved RoBERTa model to {MODEL_SAVE_PATH}")