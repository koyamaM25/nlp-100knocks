from gensim.models import KeyedVectors
import os

vec_path = "/home/koyama/nlp-100knocks/6章/code/GoogleNews-vectors-negative300.bin.gz"
output_path = "/home/koyama/nlp-100knocks/6章/out"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "out_50.txt") 

print("Loading model... s")
model = KeyedVectors.load_word2vec_format(vec_path, binary=True)

# United_States のベクトルを取得
target = "United_States"
vec = model[target]

print(f"Vector for {target}:")
print(vec)
print(f"Length = {len(vec)}")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Vector for {target} ({len(vec)} dims)\n")
    f.write(" ".join(map(str, vec.tolist())) + "\n")

