import pandas as pd

# 54番で出力したファイルのパス
input_file = "/home/koyama/nlp-100knocks/6章/out/out_54.csv"
output_file = "/home/koyama/nlp-100knocks/6章/out/out_55.txt"

semantic_correct = 0
semantic_total = 0
syntactic_correct = 0
syntactic_total = 0

# 現在のカテゴリを記録する変数
is_syntactic_category = False

with open(input_file, "r", encoding="utf-8") as f:
    # 1行目はヘッダーなのでスキップ
    next(f)
    
    for line in f:
        line = line.strip()
        if not line:
            continue
            
        # 1. カテゴリ行の判定（54番で # を先頭につけたため）
        if line.startswith("#"):
            # "gram" が含まれていれば文法タスクとみなす
            if "gram" in line:
                is_syntactic_category = True
            else:
                is_syntactic_category = False
            continue

        # 2. データ行の処理
        parts = line.split(',')
        
        # CSVの列数が足りない行や、OOV（語彙外）の行はスキップ
        if len(parts) < 6 or parts[4] == "OOV":
            continue
            
        # parts = [a, b, c, d_gold, pred_word, score]
        d_gold = parts[3]
        pred_word = parts[4]
        
        # 正解判定
        is_correct = (d_gold == pred_word)
        
        # 集計
        if is_syntactic_category:
            syntactic_total += 1
            if is_correct:
                syntactic_correct += 1
        else:
            semantic_total += 1
            if is_correct:
                semantic_correct += 1

# 正解率の計算
semantic_acc = semantic_correct / semantic_total if semantic_total > 0 else 0
syntactic_acc = syntactic_correct / syntactic_total if syntactic_total > 0 else 0
overall_acc = (semantic_correct + syntactic_correct) / (semantic_total + syntactic_total)

print(f"Overall accuracy: {overall_acc:.4f}")
print(f"Semantic accuracy: {semantic_acc:.4f}")
print(f"Syntactic accuracy: {syntactic_acc:.4f}")

# ファイル出力
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Overall accuracy: {overall_acc:.4f}\n")
    f.write(f"Semantic accuracy: {semantic_acc:.4f}\n")
    f.write(f"Syntactic accuracy: {syntactic_acc:.4f}\n")