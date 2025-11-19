import os
import re
import requests

input_file = "out/out_20.txt"   # イギリス記事本文
os.makedirs("out", exist_ok=True)
output_file = os.path.join("out", "out_29.txt")

# --- 基礎情報テンプレート抽出 ---
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

m = re.search(
    r'^\{\{基礎情報.*?\n(.*?)^\}\}$',
    text,
    flags=re.MULTILINE | re.DOTALL
)
if not m:
    raise ValueError("基礎情報テンプレートが見つかりませんでした")

template_body = m.group(1)

pattern = r'''
^ \|
([^=]+?)            # フィールド名
\s*=\s*
(.*?)               # 値
(?=
    \n\|[^=]+?=     # 次のフィールド
  | \n\}\}          # テンプレ終了
)
'''

fields = re.findall(
    pattern,
    template_body,
    flags=re.MULTILINE | re.DOTALL | re.VERBOSE
)

basic_info = {name.strip(): value.strip() for name, value in fields}

# --- 国旗画像ファイル名取得 ---
flag_filename = basic_info.get("国旗画像")
if flag_filename is None:
    raise KeyError("『国旗画像』フィールドがありません")

# --- MediaWiki API を叩いてURL取得 ---
def get_flag_image_url(filename: str) -> str:
    endpoint = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": f"File:{filename}",
        "iiprop": "url",
    }
    headers = {
        # 自分を識別できるUser-Agentを付ける（プロジェクト名やGitHub ID, メール等）
        "User-Agent": "nlp-100knocks/0.1 (koyamaM25; contact@example.com)"
    }

    r = requests.get(endpoint, params=params, headers=headers, timeout=10)
    r.raise_for_status()  # ここで403なら例外が出る

    data = r.json()

    pages = data.get("query", {}).get("pages", {})
    for pageid, page in pages.items():
        info = page.get("imageinfo")
        if not info:
            continue
        return info[0].get("url")

    raise ValueError("画像URLが取得できませんでした")


flag_url = get_flag_image_url(flag_filename)

# --- ファイル出力 ---
with open(output_file, "w", encoding="utf-8") as out:
    out.write(flag_url + "\n")

print("URL:", flag_url)
