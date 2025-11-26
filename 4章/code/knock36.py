import gzip
import json
import os
import re
from collections import Counter

import spacy
from spacy.tokens import Doc

patterns = [
    # 強調（'' ～ '''''） → 削除
    (re.compile(r"'{2,5}"), ""),

    # 内部リンク [[記事名|表示名]] / [[表示名]] → 表示名だけ残す
    (re.compile(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]"), r"\1"),

    # 外部リンク [http://example.com 表示名] → 表示名だけ残す
    (re.compile(r"\[(https?://[^\s]+)\s+([^\]]+)\]"), r"\2"),

    # <ref> ～ </ref> を含むタグごと削除
    (re.compile(r"<ref[^>]*>.*?</ref>", flags=re.DOTALL), ""),

    # その他のHTMLタグ削除
    (re.compile(r"<[^>]+>"), ""),

    # テンプレート {{ ... }} をざっくり削除（ネストは考えない簡易版）
    (re.compile(r"\{\{.*?\}\}", flags=re.DOTALL), ""),
]


def remove_markup(text: str) -> str:
    """WikipediaのMediaWikiマークアップをある程度ざっくり除去する."""
    for pat, repl in patterns:
        text = pat.sub(repl, text)
    return text


def split_text_by_bytes(text: str, max_bytes: int = 49149) -> list[str]:
    """
    Sudachi の制限（<= 49149 bytes）に合わせてテキストを分割する。
    UTF-8 バイト長で分割し、各チャンクを文字列に戻して返す。
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return [text]

    chunks: list[str] = []
    for i in range(0, len(encoded), max_bytes):
        chunk_bytes = encoded[i : i + max_bytes]
        # 分割位置で文字が途中になる可能性があるので、ignore で復元
        chunk = chunk_bytes.decode("utf-8", errors="ignore")
        if chunk:  # 空でなければ追加
            chunks.append(chunk)
    return chunks


def main():
    input_path = "/home/koyama/nlp-100knocks/4章/code/jawiki-country.json.gz"

    out_dir = "/home/koyama/nlp-100knocks/4章/out"
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, "out_36.txt")

    nlp = spacy.load("ja_ginza")

    # 単語頻度カウンタ
    counter = Counter()

    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            text = article.get("text", "")

            cleaned = remove_markup(text)

            chunks = split_text_by_bytes(cleaned)

            for chunk in chunks:
                # 空白だけのチャンクはスキップ
                if not chunk.strip():
                    continue

                doc: Doc = nlp(chunk)

                for token in doc:
                    # 空白や句読点などはスキップ
                    if token.is_space or token.is_punct:
                        continue

                    # 原形ベースでカウント
                    surface = token.lemma_
                    if surface:
                        counter[surface] += 1

    top20 = counter.most_common(20)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for word, freq in top20:
            out_f.write(f"{word}\t{freq}\n")


if __name__ == "__main__":
    main()
