file_name = 'popular-names.txt'

n = 10

with open(file_name, 'r', encoding='utf-8') as f:
    print(f"--- ファイル '{file_name}' の先頭{n}行 (タブをスペースに置換) ---")
    
    for i in range(n):
        line = f.readline()
        if not line:
            break
        replaced_line = line.replace('\t', ' ')
        print(replaced_line, end='')

# $ head -n 10 popular-names.txt | tr '\t' ' '
# Mary F 7065 1880
# Anna F 2604 1880
# Emma F 2003 1880
# Elizabeth F 1939 1880
# Minnie F 1746 1880
# Margaret F 1578 1880
# Ida F 1472 1880
# Alice F 1414 1880
# Bertha F 1320 1880
# Sarah F 1288 1880