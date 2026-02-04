import pandas as pd
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset # Hugging Face標準のデータセットライブラリ
import evaluate # 評価指標計算用

# パス設定
OUTPUT_PATH = "/home/koyama/nlp-100knocks/9章/out"
MODEL_DIR = os.path.join(OUTPUT_PATH, 'model_trainer_api')
os.makedirs(OUTPUT_PATH, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU0のみ使用
torch.cuda.set_device(0)

# 1. データの読み込み
df_train = pd.read_pickle(os.path.join(OUTPUT_PATH, 'out_85_train.pkl'))
df_dev = pd.read_pickle(os.path.join(OUTPUT_PATH, 'out_85_dev.pkl'))

# 2. Pandas DataFrame を Hugging Face Dataset に変換
# これだけでTrainerで扱える形式になります
train_dataset = Dataset.from_pandas(df_train[['sentence', 'label']])
dev_dataset = Dataset.from_pandas(df_dev[['sentence', 'label']])

# 3. 最大長の計算（あなたのロジックをそのまま流用）
def search_maxlength(tokens_series):
    max_length = 0
    for tokens in tokens_series:
        if len(tokens) > max_length:
            max_length = len(tokens)
    return max_length

max_len_train = search_maxlength(df_train['tokens'])
max_len_dev = search_maxlength(df_dev['tokens'])
final_max_length = max(max_len_train, max_len_dev) + 2
print(f"Max length defined as: {final_max_length}")

# 4. トークナイズ処理
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    # バッチ処理で一気にトークン化
    return tokenizer(
        examples["sentence"], 
        padding="max_length", 
        truncation=True, 
        max_length=final_max_length
    )

# mapを使って全データに適用
train_dataset = train_dataset.map(preprocess_function, batched=True)
dev_dataset = dev_dataset.map(preprocess_function, batched=True)

# 【重要】Trainerは 'labels' という列名を期待するのでリネーム
train_dataset = train_dataset.rename_column("label", "labels")
dev_dataset = dev_dataset.rename_column("label", "labels")

# 不要な列（元の文など）を削除し、PyTorchのTensor型に変換
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 5. モデルの準備
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 6. 評価指標の定義
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# 7. 学習設定 (TrainingArguments)
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",       
    logging_strategy="epoch",    
    load_best_model_at_end=True,
    save_total_limit=1,
)

# 8. Trainerの初期化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

# 9. 実行
print("Starting training...")
trainer.train()

# 10. 保存
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"Model saved to {MODEL_DIR}")