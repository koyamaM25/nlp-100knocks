from gensim.models import KeyedVectors
import numpy as np
import pickle
import os

VEC_PATH = "/home/koyama/nlp-100knocks/6章/code/GoogleNews-vectors-negative300.bin.gz"
OUTPUT_PATH = "/home/koyama/nlp-100knocks/8章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading model... ")
model = KeyedVectors.load_word2vec_format(VEC_PATH, binary=True)

#.vector.shapeでmodel内のベクトル集合の次元数を取得できる
#[0]で行数（単語数）[1]で列数（次元）抽出
vocab_size = model.vectors.shape[0]  # 3000000 (単語数)
emb_dim = model.vectors.shape[1]     # 300 (次元数)

#パディングベクトルの作成
padding_vec = np.zeros((1, emb_dim))

#パディングベクトルと事前学習済みベクトル集合の垂直結合
emb_matrix = np.vstack((padding_vec, model.vectors)).astype(np.float32)

#単語とトークンIDの対応表の作成
word2id = {'<PAD>': 0}
for word, original_id in model.key_to_index.items():
    word2id[word] = original_id + 1

#保存
np.save(os.path.join(OUTPUT_PATH, 'out_70_embedding_matrix.npy'), emb_matrix)

with open(os.path.join(OUTPUT_PATH, 'out_70_word2id.pkl'), 'wb') as f:
    pickle.dump(word2id, f)