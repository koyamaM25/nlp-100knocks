import math
import os

file_name = 'popular-names.txt'
output_dir = 'out_15'
output_prefix = 'split_' # 出力ファイルの接頭辞

n = int(input("分割数を指定してください:"))
os.makedirs(output_dir, exist_ok=True)

with open(file_name, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
    total_lines = len(all_lines)
    lines_per_file = math.ceil(total_lines / n)
    
print(f"全行数: {total_lines}行を{n}分割します。")
print(f"1ファイルあたりの最大行数: {lines_per_file}行")
    
    
for i in range(n):
    start_index = i * lines_per_file
    end_index = (i + 1) * lines_per_file    
    chunk = all_lines[start_index:end_index]
    
    output_file_name = f"{output_prefix}{i:02}"
    output_path = os.path.join(output_dir, output_file_name)
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.writelines(chunk)
            
    print(f"ファイル {output_file_name} に {len(chunk)} 行を書き出しました。")

# $ split -n l/10 popular-names.txt out_15/split_