# #めちゃ時間かかる
# capital以外もやった
from gensim.models import KeyedVectors
import os

vec_path = "/home/koyama/nlp-100knocks/6章/code/GoogleNews-vectors-negative300.bin.gz"
capital_path = "/home/koyama/nlp-100knocks/6章/code/questions-words.txt"
output_path = "/home/koyama/nlp-100knocks/6章/out"

os.makedirs(output_path, exist_ok=True)
output_file = os.path.join(output_path, "out_54.csv") 

print("Loading model... ")
model = KeyedVectors.load_word2vec_format(vec_path, binary=True)

with open(capital_path, "r", encoding="utf-8") as input_f, \
     open(output_file, "w", encoding="utf-8") as output_f:

    output_f.write("a,b,c,d_gold,pred_word,score\n")

    for line in input_f:
        line = line.strip()
        if not line:
            continue

        words = line.split()

        # セクション行はそのままコメントとして残す
        if words[0].startswith(":"):
            # CSVなのでコメント扱いで出したいので先頭に # を付ける
            output_f.write(f"# {line}\n")
            continue

        # 通常行: a b c d の4語を想定
        if len(words) < 4:
            continue

        a, b, c, d_gold = words[:4]

        try:
            most_similar = model.most_similar(
                positive=[b, c],
                negative=[a],
                topn=1
            )
        except KeyError:
            # 語彙外がある場合は"OOV" として出力
            output_f.write(f"{a},{b},{c},{d_gold},OOV,OOV\n")
            continue

        pred_word, score = most_similar[0]

        output_f.write(f"{a},{b},{c},{d_gold},{pred_word},{score}\n")