import os
import re

input_file = "out/out_20.txt"
os.makedirs("out", exist_ok=True)
output_file = os.path.join("out", "out_28.txt")

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

def clean_value(value: str) -> str:
    """基礎情報の値から MediaWiki マークアップをできるだけ除去してテキストにする"""

    # 前後の空白をざっくり削る
    value = value.strip()

    # (1) 強調マークアップの除去: '' / ''' / ''''' など
    value = re.sub(r"'{2,5}", "", value)

    # (2) 内部リンクマークアップの除去
    # [[記事名]] → 記事名
    # [[記事名|表示テキスト]] → 表示テキスト
    value = re.sub(
        r'\[\[(?:[^|\]]+\|)?([^|\]]+)\]\]',
        r'\1',
        value
    )

    # (3) 外部リンクマークアップの除去
    # [http://example.com 説明] → 説明
    # [http://example.com] → 空文字 or URL（ここでは URL を消してしまう）
    # 説明テキストがあればそちらを残す
    def repl_external(m):
        label = m.group(1)
        return label if label else ""
    value = re.sub(
        r'\[https?:\/\/[^\s\]]+(?:\s+([^\]]+))?\]',
        repl_external,
        value
    )

    # (4) 言語テンプレート {{lang|en|Text}} → Text にする
    # 第3引数（あるいは最後の引数）を残すイメージ
    value = re.sub(
        r'\{\{lang\|[^|}]+\|([^|}]+)\}\}',
        r'\1',
        value
    )

    # (5) よくある仮リンクテンプレート {{仮リンク|記事名|en|...}} → 記事名
    value = re.sub(
        r'\{\{仮リンク\|([^|}]+)\|[^}]*\}\}',
        r'\1',
        value
    )

    # (6) <ref>...</ref> の除去（注釈）
    value = re.sub(
        r'<ref[^>]*?>.*?</ref>',
        '',
        value,
        flags=re.DOTALL
    )
    # <ref name="..." /> のような単独タグも除去
    value = re.sub(
        r'<ref[^>]*/>',
        '',
        value
    )

    # (7) <br>, <br />, <br/> などは改行かスペースに変換
    value = re.sub(r'<br\s*/?>', '\n', value, flags=re.IGNORECASE)

    # (8) その他の HTML 風タグ（<small> など）は中身だけ残してタグを除去
    value = re.sub(r'</?[a-zA-Z][^>]*>', '', value)

    # (9) 残っている一般的なテンプレート {{...}} は中身を取るのが難しいものも多いので、
    #     ここでは安全のためマークアップごと削除してしまう
    value = re.sub(r'\{\{[^{}]*\}\}', '', value)

    # (10) 連続する空白や改行を軽く整形
    #       改行は残したいならここは好みで
    value = re.sub(r'[ \t]+', ' ', value)
    value = value.strip()

    return value

# ------------- 辞書化＋クレンジング -------------
basic_info = {}

for name, value in fields:
    name = name.strip()
    cleaned = clean_value(value)
    basic_info[name] = cleaned

# ------------- ファイルに保存 -------------
with open(output_file, "w", encoding="utf-8") as out:
    for key, value in basic_info.items():
        out.write(f"{key}\t{value}\n")

print("✅ out/out_28.txt に整形済みの国の基本情報を書き出しました")
