import pandas as pd
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset 
import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU0のみ使用
torch.cuda.set_device(0)

OUTPUT_PATH = "/home/koyama/nlp-100knocks/9章/out"
MODEL_DIR = os.path.join(OUTPUT_PATH, 'model_89_trainer_roberta') 
os.makedirs(OUTPUT_PATH, exist_ok=True)

MODEL_NAME = "roberta-base"

# データの読み込み
df_train = pd.read_pickle(os.path.join(OUTPUT_PATH, 'out_85_train.pkl'))
df_dev = pd.read_pickle(os.path.join(OUTPUT_PATH, 'out_85_dev.pkl'))

# Pandas -> Dataset 変換
train_dataset = Dataset.from_pandas(df_train[['sentence', 'label']])
dev_dataset = Dataset.from_pandas(df_dev[['sentence', 'label']])

# 最大長の計算
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

# トークナイザと前処理
# RoBERTaのトークナイザを読み込みます
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(
        examples["sentence"], 
        padding="max_length", 
        truncation=True, 
        max_length=final_max_length
    )

# 一括処理
print("Tokenizing data...")
train_dataset = train_dataset.map(preprocess_function, batched=True)
dev_dataset = dev_dataset.map(preprocess_function, batched=True)

# 列名変更 (label -> labels)
train_dataset = train_dataset.rename_column("label", "labels")
dev_dataset = dev_dataset.rename_column("label", "labels")

# Tensor形式に変換
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# モデルの準備
print(f"Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 評価指標
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# 学習設定 
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    learning_rate=2e-5,
    fp16=True,                  
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=32,
    dataloader_num_workers=0,   
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",      
    save_strategy="epoch",      
    logging_strategy="epoch",   
    load_best_model_at_end=True,
    save_total_limit=1,
)

# Trainer起動
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

# 学習開始
print("Starting training with RoBERTa...")
trainer.train()

# 保存
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"RoBERTa Model saved to {MODEL_DIR}")

# 最終評価
print("\n=== Final Evaluation ===")

# Dev評価
metrics_dev = trainer.evaluate()
print(f"Dev Accuracy:   {metrics_dev['eval_accuracy']:.4f}")

# Train評価
metrics_train = trainer.evaluate(train_dataset)
print(f"Train Accuracy: {metrics_train['eval_accuracy']:.4f}")

print("-" * 30)
print(f"Model: {MODEL_NAME}")
print(f"Train Acc: {metrics_train['eval_accuracy']:.4f}")
print(f"Dev Acc:   {metrics_dev['eval_accuracy']:.4f}")
print("-" * 30)