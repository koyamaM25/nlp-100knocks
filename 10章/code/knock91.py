from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

OUTPUT_PATH = "/home/koyama/nlp-100knocks/10章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'out_91.txt')

text = "The movie was full of"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer(text, return_tensors='pt')


tempatures = [0.5, 0.7, 1.0, 1.5, 2.0]
outputs = []
# 推論
model.eval()
with torch.no_grad():
    for temp in tempatures:
        output = model.generate(
            **inputs, 
            do_sample=True, 
            temperature=temp, 
            max_new_tokens=50, 
            num_return_sequences=5
            )
        outputs.append(output)


with open(OUTPUT_FILE, 'w', encoding='UTF-8') as f:
    for i, temp in enumerate(tempatures):
        f.write(f"Temperature: {temp}\n")
        for j in range(5):
            generated_text = tokenizer.decode(outputs[i][j], skip_special_tokens=True)
            f.write(f"{j+1}: {generated_text}\n")
        f.write("\n")