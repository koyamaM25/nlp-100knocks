import gzip
import json

def load_uk_text(path: str) -> str:
    """
    jawiki-country.json(.gz) から「イギリス」記事の本文を取り出して返す
    """
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            if article.get('title') == 'イギリス':
                return article.get('text')

    raise ValueError('「イギリス」記事が見つかりませんでした')

path = 'jawiki-country.json.gz'

uk_text = load_uk_text(path)

print(uk_text)

