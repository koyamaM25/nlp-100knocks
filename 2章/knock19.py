file_name = 'popular-names.txt'
output_file = 'out_19.txt'

def get_sort_key(line):
    """行をタブで分割し、3列目（インデックス2）の値を整数に変換して返す"""
    try:
        return int(line.split('\t')[2])
    except:
        return 0

with open(file_name, 'r', encoding='utf-8') as f:
    # 全行を読み込む
    lines = f.readlines()
    
print(f"--- ファイル '{file_name}' の3列目を数値降順で整列 ---")

sorted_lines = sorted(lines, key=get_sort_key, reverse=True)

with open(output_file, 'w', encoding='utf-8') as out_f:
    out_f.writelines(sorted_lines)

# $ sort -k 3,3nr popular-names.txt > out_19.txt