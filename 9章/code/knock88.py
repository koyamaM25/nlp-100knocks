from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os

OUTPUT_PATH = "/home/koyama/nlp-100knocks/9章/out"
MODEL_DIR = os.path.join(OUTPUT_PATH, 'model_87')
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'out_88.txt')

texts = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

# モデルとトークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

with open(OUTPUT_FILE, 'w', encoding='UTF-8')as f:
    f.write("\n" + "="*30 + "\n")
    f.write("     Inference Results\n")
    f.write("="*30 + "\n")
    for text in texts:
        # トークン化
        # return_tensors="pt" でPyTorch形式に
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # データをデバイスへ移動
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 推論
            outputs = model(**inputs)
            logits = outputs.logits # モデルの生の出力スコア
            
            # 確率に変換 (Softmax)
            probs = F.softmax(logits, dim=-1)
            
            # 最も確率が高いクラスを選ぶ
            pred_label = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][pred_label].item()

        # --- 3. 結果の表示 ---
        # SST-2データセットの定義: 0=Negative, 1=Positive
        label_name = "Positive" if pred_label == 1 else "Negative"
        
    
        f.write(f"Text:   {text}\n")
        f.write(f"Result: {label_name} (Label: {pred_label})\n")
        f.write(f"Score:  {confidence:.4f}\n")
        f.write("-" * 30 + "\n")