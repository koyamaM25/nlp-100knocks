import zipfile
import pandas as pd

input_file = "/home/koyama/nlp-100knocks/7章/code/SST-2.zip"

with zipfile.ZipFile(input_file, "r") as z:
    with z.open('SST-2/train.tsv') as f:
        df_train = pd.read_csv(f, sep='\t')
    with z.open('SST-2/dev.tsv') as f:
        df_dev = pd.read_csv(f, sep='\t')
    
print(f"学習データ")
print(f"{df_train['label'].value_counts()}\n")

print(f"検証データ")
print(f"{df_dev['label'].value_counts()}\n")
