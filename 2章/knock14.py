file_name = 'popular-names.txt'

n = 10

with open(file_name, 'r', encoding='utf-8') as f:
    print(f"--- ファイル '{file_name}' の先頭{n}行の1列目 ---")
    
    for i in range(n):
        line = f.readline()
        if not line:
            break
        
        line = line.split('\t')
        if line:
            print(line[0])

# $ head -n 10 popular-names.txt | cut -f 1 -d $'\t'
# Mary
# Anna
# Emma
# Elizabeth
# Minnie
# Margaret
# Ida
# Alice
# Bertha
# Sarah