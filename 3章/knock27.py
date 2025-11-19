import os
import re

input_file = "out/out_20.txt"
os.makedirs("out", exist_ok=True)
output_file = os.path.join("out", "out_27.txt")

# ------------- 基礎情報テンプレート抽出 -------------
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# {{基礎情報 ... }} の中身だけ取り出す
m = re.search(
    r'^\{\{基礎情報.*?\n(.*?)^\}\}$',
    text,
    flags=re.MULTILINE | re.DOTALL
)
if not m:
    raise ValueError("基礎情報テンプレートが見つかりませんでした")

template_body = m.group(1)

# ------------- 各フィールド抽出 -------------
pattern = r'''
^ \|
([^=]+?)            # フィールド名
\s*=\s*
(.*?)               # 値（次のフィールド直前まで）
(?=
    \n\|[^=]+?=     # 次のフィールド
  | \n\}\}          # テンプレート終了
)
'''

fields = re.findall(
    pattern,
    template_body,
    flags=re.MULTILINE | re.DOTALL | re.VERBOSE
)

basic_info = {}

for name, value in fields:
    name = name.strip()
    value = value.strip()

    # ------------- (1) 強調マークアップの除去 -------------
    # ''（弱い強調）, '''（強調）, '''''（強い強調）など
    value = re.sub(r"'{2,5}", "", value)

    # ------------- (2) 内部リンクマークアップの除去 -------------
    # [[記事名]] → 記事名
    # [[記事名|表示テキスト]] → 表示テキスト
    value = re.sub(
        r'\[\[(?:[^|\]]+\|)?([^|\]]+)\]\]',
        r'\1',
        value
    )

    basic_info[name] = value

# ------------- ファイルに保存 -------------
with open(output_file, "w", encoding="utf-8") as out:
    for key, value in basic_info.items():
        out.write(f"{key}\t{value}\n")

