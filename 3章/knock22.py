import os
import re

input_file = "out/out_20.txt"
os.makedirs("out", exist_ok=True)
output_file = os.path.join("out", "out_22.txt")

with open(input_file, "r", encoding="utf-8") as input_f, \
     open(output_file, "w", encoding="utf-8") as output_f:

    text = input_f.read()#1つの文字列として読み込み

    # [[Category:名前]] または [[Category:名前|ソートキー]] にマッチ
    pattern = r"\[\[Category:([^|\]]+)"#| と ] 以外の文字が1文字以上続くもの
    names = re.findall(pattern, text)#text 全体から pattern にマッチする キャプチャ部分（丸括弧の中）をリストとして全部集める

    # 前後の空白を削って重複除去
    names = sorted(set(name.strip() for name in names))

    for name in names:
        output_f.write(name + "\n")
