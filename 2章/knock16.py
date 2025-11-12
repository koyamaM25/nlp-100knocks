import random

file_name = 'popular-names.txt'

with open(file_name, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
print(f"--- ファイル '{file_name}' の行をランダムに並び替え ---")

random.shuffle(lines)
    
with open('out_16.txt', 'w', encoding='utf-8') as out_f:
    out_f.writelines(lines)

#$ shuf popular-names.txt > out_16.txt