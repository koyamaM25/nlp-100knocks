from gensim.models import KeyedVectors
import os

vec_path = "/home/koyama/nlp-100knocks/6章/code/GoogleNews-vectors-negative300.bin.gz"
output_path = "/home/koyama/nlp-100knocks/6章/out"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "out_53.txt") 

print("Loading model... ")
model = KeyedVectors.load_word2vec_format(vec_path, binary=True)

# Spain のベクトルを取得
target = "Spain"

new_vec = model[target] - model["Madrid"] + model["Athens"]
most_similar_new_vec10 = model.most_similar(positive=[new_vec], topn=10)

print(f"Vector for {target} ")
print(f"Analogy: {target} - Madrid + Athens")
for word, score in most_similar_new_vec10:
    print(word, score)

with open(output_file, "w", encoding="utf-8") as f:
    f.write("Spain - Madrid + Athens\n")
    for word, score in most_similar_new_vec10:
        f.write(f"{word}\t{score}\n")