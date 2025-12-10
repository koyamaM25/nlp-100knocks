from gensim.models import KeyedVectors
import os

vec_path = "/home/koyama/nlp-100knocks/6章/code/GoogleNews-vectors-negative300.bin.gz"
output_path = "/home/koyama/nlp-100knocks/6章/out"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "out_52.txt") 

print("Loading model... ")
model = KeyedVectors.load_word2vec_format(vec_path, binary=True)

# United_States のベクトルを取得
target = "United_States"

most_similar10 = model.most_similar(target, topn=10)

print(f"Vector for {target} ")
for word, score in most_similar10:
    print(f"{word}\t{score}")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Vector for {target}\n")
    for word, score in most_similar10:
        f.write(f"{word}\t{score}\n")
