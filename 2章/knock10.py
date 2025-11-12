file_name = 'popular-names.txt'

line_count = 0
with open(file_name, 'r', encoding='utf-8') as f:
    for line in f:
        line_count += 1
    
print(f"ファイルの行数: {line_count}")

# $ wc -l popular-names.txt
# 2780 popular-names.txt