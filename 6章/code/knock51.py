from gensim.models import KeyedVectors
import os

vec_path = "/home/koyama/nlp-100knocks/6章/code/GoogleNews-vectors-negative300.bin.gz"
output_path = "/home/koyama/nlp-100knocks/6章/out"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "out_51.txt") 

print("Loading model... ")
model = KeyedVectors.load_word2vec_format(vec_path, binary=True)

# United_States U.S のベクトルを取得
target1 = "United_States"
target2 = "U.S"

similarity = model.similarity(target1,target2)

print(f"Vector for {target1} {target2}")
print(f"similarity : {similarity}")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Vector for {target1} {target2}\n")
    f.write(f"similarity : {similarity}\n")
