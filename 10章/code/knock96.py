from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import pandas as pd
import zipfile
import os


OUTPUT_PATH = "/home/koyama/nlp-100knocks/10章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "out_96.txt")
SST2_PATH = "/home/koyama/nlp-100knocks/7章/code/SST-2.zip"

MODEL_NAME = "gpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def load_sst2_data():
    with zipfile.ZipFile(SST2_PATH, "r") as z:
        with z.open('SST-2/train.tsv') as f:
            df_train = pd.read_csv(f, sep='\t')
        with z.open('SST-2/dev.tsv') as f:
            df_dev = pd.read_csv(f, sep='\t')
    return {"train": df_train, "dev": df_dev}

def make_prompt(text):
    return (
        f"Classify the sentiment of the movie review as positive or negative.\n"
        f"Only reply with a single word: \"positive\" or \"negative\".\n"
        f"Sentence: {text}\n"
        f"Answer:"
    )

def make_fewshot_prompt(text):
    return (
    f"Classify the sentiment of the movie review as positive or negative.\n"
    f"Only reply with a single word: \"positive\" or \"negative\".\n"

    f"Examples:\n"
    f"Sentence: I love this movie! It's amazing.\n"
    f"Answer: positive\n"

    f"Sentence: The plot was boring and predictable.\n"
    f"Answer: negative\n"

    f"Sentence: The acting was fantastic, I enjoyed it a lot.\n"
    f"Answer: positive\n"

    f"Sentence: I did not like the food at all.\n"
    f"Answer: negative\n"

    f"Now classify this sentence:\n"
    f"Sentence: {text}\n"
    f"Answer:"
    )

def predict_label(text):
    prompt = make_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_tokens = outputs[0, inputs["input_ids"].shape[1]:] 
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()

    if "positive" in answer:
        pred = 1
    elif "negative" in answer:
        pred = 0
    else:
        pred = 0

    return pred, answer


def predict_label_fewshot(text):
    prompt_fewshot = make_fewshot_prompt(text)
    inputs_fewshot = tokenizer(prompt_fewshot, return_tensors="pt").to(DEVICE)

    outputs_fewshot = model.generate(
        **inputs_fewshot,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_tokens = outputs_fewshot[0, inputs_fewshot["input_ids"].shape[1]:] 
    answer_fewshot = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()

    if "positive" in answer_fewshot:
        pred = 1
    elif "negative" in answer_fewshot:
        pred = 0
    else:
        pred = 0

    return pred, answer_fewshot

df_dev = load_sst2_data()["dev"]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    # zero-shot
    correct = 0
    f.write("=== Zero-shot (generate) ===\n\n")

    for i, (text, label) in enumerate(zip(df_dev["sentence"], df_dev["label"])):
        pred, answer = predict_label(text)
        correct += int(pred == int(label))

        # 先頭5件だけ中身を保存
        if i < 5:
            f.write(f"[{i}]\n")
            f.write(f"Sentence: {text}\n")
            f.write(f"Gold: {int(label)}\n")
            f.write(f"Pred: {pred}\n")
            f.write(f"Generated: {answer}\n\n")

    accuracy = correct / len(df_dev)
    f.write(f"Accuracy: {accuracy:.3f}\n\n")

    # few-shot
    correct_fewshot = 0
    f.write("=== Few-shot (generate) ===\n\n")

    for i, (text, label) in enumerate(zip(df_dev["sentence"], df_dev["label"])):
        pred_fewshot, answer_fewshot = predict_label_fewshot(text)
        correct_fewshot += int(pred_fewshot == int(label))

        if i < 5:
            f.write(f"[{i}]\n")
            f.write(f"Sentence: {text}\n")
            f.write(f"Gold: {int(label)}\n")
            f.write(f"Pred: {pred_fewshot}\n")
            f.write(f"Generated: {answer_fewshot}\n\n")

    accuracy_fewshot = correct_fewshot / len(df_dev)
    f.write(f"Accuracy few-shot: {accuracy_fewshot:.3f}\n")