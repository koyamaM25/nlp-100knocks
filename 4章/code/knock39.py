import gzip
import json
import os
import re
from collections import Counter

import math
import spacy
import matplotlib.pyplot as plt

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
        if chunk:
            chunks.append(chunk)
    return chunks


def main():
    input_path = "/home/koyama/nlp-100knocks/4章/code/jawiki-country.json.gz"

    out_dir = "/home/koyama/nlp-100knocks/4章/out"
    os.makedirs(out_dir, exist_ok=True)
    freq_file = os.path.join(out_dir, "out_39.txt")
    fig_file = os.path.join(out_dir, "out_39.png")

    nlp = spacy.load("ja_ginza")

    counter = Counter()

    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            text: str = article.get("text", "")

            cleaned = remove_markup(text)

            chunks = split_text_by_bytes(cleaned)

            for chunk in chunks:
                if not chunk.strip():
                    continue

                doc = nlp(chunk)

                for token in doc:
                    # 空白・句読点をスキップ
                    if token.is_space or token.is_punct:
                        continue

                    # ここでは原形でカウント（表層形にしたければ token.text）
                    surface = token.lemma_
                    if surface:
                        counter[surface] += 1

    # 頻度の降順に並べて，順位と頻度のリストを作る
    # counter.most_common() は (単語, 頻度) を頻度降順に返す
    freq_list = [freq for _, freq in counter.most_common()]
    ranks = list(range(1, len(freq_list) + 1))

    with open(freq_file, "w", encoding="utf-8") as out_f:
        out_f.write("word\tfreq\n")
        for word, freq in counter.most_common(20):  # 上位100語だけ
            out_f.write(f"{word}\t{freq}\n")

    # 両対数グラフをプロット
    plt.figure(figsize=(10, 6))

    # 散布図：点でプロット
    plt.scatter(ranks, freq_list, s=10, alpha=0.6)

    # 両対数スケール
    plt.xscale("log")
    plt.yscale("log")

    # タイトル・ラベル
    plt.title("Zipf's Law (Log-Log Plot)")
    plt.xlabel("Rank (log scale)")
    plt.ylabel("Frequency (log scale)")

    # グリッド（メジャー・マイナー両方）
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(fig_file)


if __name__ == "__main__":
    main()
