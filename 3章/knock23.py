import os
import re

input_file = "out/out_20.txt"
os.makedirs("out", exist_ok=True)
output_file = os.path.join("out", "out_23.txt")

pattern = r'^(=+)\s*(.+?)\s*\1\s*$'
#行頭 ^ から始まって、= が 1 個以上連続する部分をキャプチャ
#\s 0 個以上の空白
#(.+?) 見出しの中身（セクション名）を最短一致でキャプチャ
#\1 最初の括弧 (=+) にマッチしたものと同じ文字列

with open(input_file, "r", encoding="utf-8") as input_f, \
     open(output_file, "w", encoding="utf-8") as output_f:

    for line in input_f:
        line = line.rstrip('\n')
        m = re.match(pattern, line)
        if m:
            eqs = m.group(1)      # 先頭の ====... の部分
            name = m.group(2)     # セクション名
            level = len(eqs) - 1  # == ... == → 1, === ... === → 2, ...

            output_f.write(f"{level}\t{name}\n")
