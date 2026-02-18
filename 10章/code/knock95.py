from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# =========================
# Paths
# =========================
OUTPUT_PATH = "/home/koyama/nlp-100knocks/10章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "out_95.txt")

# =========================
# Model / Tokenizer
# =========================
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-4B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")

def run_chat(messages, max_new_tokens=128):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")  # device_mapに任せる

    with torch.no_grad():
        out_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    gen_only = out_ids[0, model_inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True).strip()

# =========================
# Multi-turn chat
# =========================
prompt1 = "What do you call a sweet eaten after dinner?"
prompt2 = "Please give me the plural form of the word with its spelling in reverse order."

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt1},
]

# Turn 1
assistant1 = run_chat(messages, max_new_tokens=64)
messages.append({"role": "assistant", "content": assistant1})

# Turn 2
messages.append({"role": "user", "content": prompt2})
assistant2 = run_chat(messages, max_new_tokens=128)
messages.append({"role": "assistant", "content": assistant2})

with open(OUTPUT_FILE, "w", encoding="UTF-8") as f:
    f.write("=== Multi-turn Chat ===\n\n")
    f.write(f"[USER]\n{prompt1}\n\n")
    f.write(f"[ASSISTANT]\n{assistant1}\n\n")
    f.write(f"[USER]\n{prompt2}\n\n")
    f.write(f"[ASSISTANT]\n{assistant2}\n")

print(f"Wrote: {OUTPUT_FILE}")