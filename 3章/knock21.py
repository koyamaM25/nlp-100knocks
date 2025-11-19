import os
import re

input_file = "out/out_20.txt"
os.makedirs("out", exist_ok=True)
output_file = os.path.join("out", "out_21.txt")

pattern = r"\[\[Category:.*\]\]" #[[Category: * ]]

with open(input_file, "r", encoding="utf-8") as input_f, \
     open(output_file, "w", encoding="utf-8") as output_f:

    for line in input_f:
        if re.search(pattern, line):
            output_f.write(line)
