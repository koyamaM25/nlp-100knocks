from transformers import AutoTokenizer
import zipfile
import pandas as pd
import pickle
import os

SST_DATA = "/home/koyama/nlp-100knocks/7章/code/SST-2.zip"
OUTPUT_PATH = "/home/koyama/nlp-100knocks/9章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)

with zipfile.ZipFile(SST_DATA, "r") as z:
    with z.open('SST-2/train.tsv') as f:
        df_train = pd.read_csv(f, sep='\t')
    with z.open('SST-2/dev.tsv') as f:
        df_dev = pd.read_csv(f, sep='\t')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_texts(df):
    df['tokens'] = df['sentence'].apply(tokenizer.tokenize)
    return df

df_dev = tokenize_texts(df_dev)
df_train = tokenize_texts(df_train)
print(df_dev[:1])
print(df_train[:1])

# 保存
df_train.to_pickle(os.path.join(OUTPUT_PATH, 'out_85_train.pkl'))
df_dev.to_pickle(os.path.join(OUTPUT_PATH, 'out_85_dev.pkl'))
