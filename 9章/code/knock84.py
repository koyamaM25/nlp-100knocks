from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os

OUTPUT_PATH = "/home/koyama/nlp-100knocks/9章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'out_84.txt')

texts = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

embs = []
# 単語→ID
for text in texts:
    inputs = tokenizer(text, return_tensors='pt')
    # 推論
    with torch.no_grad():
        outputs = model(**inputs)
        embs.append(outputs.last_hidden_state[0,0,:])
        # dim=0 で「単語数」の方向につぶして平均をとる
        token_mean = outputs.last_hidden_state[0].mean(dim=0)
        embs.append(token_mean)

with open(OUTPUT_FILE, 'w', encoding='UTF-8') as f:
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = F.cosine_similarity(embs[i], embs[j], dim=0)
            f.write(f"{texts[i]}\t{texts[j]}\t{sim.item():.4f}\n")