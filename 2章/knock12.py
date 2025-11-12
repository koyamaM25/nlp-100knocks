file_name = 'popular-names.txt'

n = int(input("行数を指定してください:"))

with open(file_name, 'r', encoding='utf-8') as f:
    line = f.readlines()
    tail = line[-n:]
    for lines in tail:
        print(lines.rstrip('\n'))

# $ head -n 10 popular-names.txt
# Mary    F       7065    1880
# Anna    F       2604    1880
# Emma    F       2003    1880
# Elizabeth       F       1939    1880
# Minnie  F       1746    1880
# Margaret        F       1578    1880
# Ida     F       1472    1880
# Alice   F       1414    1880
# Bertha  F       1320    1880
# Sarah   F       1288    1880