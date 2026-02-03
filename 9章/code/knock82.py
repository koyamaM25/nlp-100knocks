from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F
import os

OUTPUT_PATH = "/home/koyama/nlp-100knocks/9章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'out_82.txt')

text = "The movie was full of [MASK]."

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# 単語→ID
inputs = tokenizer(text, return_tensors='pt')

mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

# 推論
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

mask_token_logits = logits[0, mask_token_index, :]

# ロジットを確率に変換 (Softmax)
mask_token_probs = F.softmax(mask_token_logits, dim=-1)

# 上位10個のトークンIDと、その確率を取得
top10 = torch.topk(mask_token_probs, 10, dim=-1)
top10_ids = top10.indices
top10_probs = top10.values

with open(OUTPUT_FILE, 'w', encoding='UTF-8') as f:
    for token_id, prob in zip(top10_ids[0], top10_probs[0]):
        # ID→単語
        word = tokenizer.decode(token_id)
        f.write(f"{word}\t{prob.item():.5f}\n")