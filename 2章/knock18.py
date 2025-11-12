from collections import Counter

output_file = 'out_18.txt'
file_name = 'popular-names.txt'
name_counts = Counter() # 1列目の文字列の出現回数を記録するCounterオブジェクト


with open(file_name, 'r', encoding='utf-8') as f:
    for line in f:
        columns = line.split('\t')
            
        if columns and columns[0]:
            name_counts[columns[0]] += 1
    
print(f"--- ファイル '{file_name}' の1列目の出現頻度ランキング ---")
    
with open(output_file, 'w', encoding='utf-8') as out_f:
    for name, count in name_counts.most_common():
        output_line = f"{count}\t{name}\n"
        out_f.write(output_line)
    
# $ cut -f 1 popular-names.txt | sort | uniq -c | sort -r > out_18.txt 
