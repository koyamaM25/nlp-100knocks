text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

import spacy
import os

os.makedirs("/home/koyama/nlp-100knocks/4章/out", exist_ok=True)
output_file = os.path.join("/home/koyama/nlp-100knocks/4章/out", "out_32.txt")

nlp = spacy.load("ja_ginza")
doc = nlp(text)

with open(output_file, "w", encoding="utf-8") as output_f:
    for i in range(len(doc)-2):
        if (doc[i].pos_ == "NOUN") and (doc[i+1].text == "の") and (doc[i+2].pos_ == "NOUN"):
            output_f.write(f"{doc[i:i+3]}\n")


