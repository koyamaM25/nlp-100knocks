from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import os

OUTPUT_PATH = "/home/koyama/nlp-100knocks/10章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'out_90.txt')

text = "The movie was full of"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer(text, return_tensors='pt')

# 推論
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 最後のトークンのロジットを取得
logits = logits[0, -1, :]

# ロジットを確率に変換 (Softmax)
next_token_probs = F.softmax(logits, dim=-1)

# 上位10個のトークンIDと、その確率を取得
top10 = torch.topk(next_token_probs, 10, dim=-1)
top10_ids = top10.indices
top10_probs = top10.values

with open(OUTPUT_FILE, 'w', encoding='UTF-8') as f:
    for i in range(10):
        f.write(f"{tokenizer.decode(top10_ids[i])}\t{top10_probs[i]}\n")