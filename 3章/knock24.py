import os
import re

input_file = "out/out_20.txt"
os.makedirs("out", exist_ok=True)
output_file = os.path.join("out", "out_24.txt")

with open(input_file, "r", encoding="utf-8") as input_f, \
     open(output_file, "w", encoding="utf-8") as output_f:

    text = input_f.read()

    # [[File:xxxx]] または [[ファイル:xxxx]] 
    pattern = r"\[\[(?:File|ファイル):([^|\]]+)"
    files = re.findall(pattern, text)

    files = sorted(set(name.strip() for name in files))

    for name in files:
        output_f.write(name + "\n")
