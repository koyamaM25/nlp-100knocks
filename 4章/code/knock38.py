import gzip
import json
import os
import re
import math
from collections import Counter

import spacy

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
    output_file = os.path.join(out_dir, "out_38.txt")

    nlp = spacy.load("ja_ginza")
    
    N_docs = 0# コーパス全体での文書数 N
    df_counter = Counter()# 全記事を通した「文書頻度」 df(w)
    tf_japan = Counter()# 「日本に関する記事」中での TF（名詞の総出現回数）

    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            title: str = article.get("title", "")
            text: str = article.get("text", "")

            N_docs += 1

            cleaned = remove_markup(text)
            chunks = split_text_by_bytes(cleaned)

            # この記事に出現した名詞（原形）のリスト
            nouns_in_article: list[str] = []

            for chunk in chunks:
                if not chunk.strip():
                    continue

                doc = nlp(chunk)

                for token in doc:
                    # 空白・句読点をスキップ
                    if token.is_space or token.is_punct:
                        continue

                    # 名詞のみ対象（普通名詞+固有名詞）
                    if token.pos_ not in ("NOUN", "PROPN"):
                        continue

                    lemma = token.lemma_
                    if lemma:
                        nouns_in_article.append(lemma)

            if not nouns_in_article:
                continue

            # ---- 文書頻度 df(w) 用の更新（コーパス全体） ----
            noun_set = set(nouns_in_article)
            for w in noun_set:
                df_counter[w] += 1

            # ---- 「日本に関する記事」なら TF に加算 ----
            if "日本" in title:
                for w in nouns_in_article:
                    tf_japan[w] += 1

    # ---- IDF の計算（日本記事で出てきた語だけでよい）----
    idf: dict[str, float] = {}
    for w in tf_japan.keys():
        df = df_counter.get(w, 0)
        if df == 0:
            continue
        idf[w] = math.log(N_docs / df)

    # ---- TF・IDF・TF-IDF をまとめる ----
    records = []  # (word, tf, idf, tfidf)
    for w, tf_val in tf_japan.items():
        if w not in idf:
            continue
        idf_val = idf[w]
        tfidf_val = tf_val * idf_val
        records.append((w, tf_val, idf_val, tfidf_val))

    # TF-IDF の降順で上位20語を取得
    records.sort(key=lambda x: x[3], reverse=True)
    top20 = records[:20]

    with open(output_file, "w", encoding="utf-8") as out_f:
        for w, tf_val, idf_val, tfidf_val in top20:
            out_f.write(f"{w}\t{tf_val}\t{idf_val:.6f}\t{tfidf_val:.6f}\n")

    for w, tf_val, idf_val, tfidf_val in top20:
        print(f"{w}\tTF={tf_val}\tIDF={idf_val:.6f}\tTF-IDF={tfidf_val:.6f}")


if __name__ == "__main__":
    main()
