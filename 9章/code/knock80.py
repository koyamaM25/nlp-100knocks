from transformers import AutoTokenizer
import os

OUTPUT_PATH = "/home/koyama/nlp-100knocks/9章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'out_80.txt')

text = "The movie was full of incomprehensibilities."

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(text)

with open(OUTPUT_FILE, 'w', encoding='UTF-8') as f:
    for token in tokens:
        f.write(f"{token}\n")