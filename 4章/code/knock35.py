text = "メロスは激怒した。"

import spacy
import os
from spacy import displacy

os.makedirs("/home/koyama/nlp-100knocks/4章/out", exist_ok=True)
output_file = os.path.join("/home/koyama/nlp-100knocks/4章/out", "out_35.svg")

nlp = spacy.load("ja_ginza")
doc = nlp(text)
svg = displacy.render(doc, style="dep")

with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"{svg}\n")
