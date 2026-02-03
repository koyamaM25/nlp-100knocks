from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
import torch

OUTPUT_PATH = "/home/koyama/nlp-100knocks/9章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'out_81.txt')

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
best_token_id = torch.argmax(mask_token_logits, dim=-1)

# ID→単語
best_word = tokenizer.decode(best_token_id)

with open(OUTPUT_FILE, 'w', encoding='UTF-8') as f:
    f.write(f"{best_word}")