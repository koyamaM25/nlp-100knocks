s = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'

s = s.replace(',', '').replace('.', '')
words = s.split(' ')

one_char_positions = {1, 5, 6, 7, 8, 9, 15, 16, 19}
result_dict = {}

for i, w in enumerate(words, start=1):
    if i in one_char_positions:
        n = 1
    else:
        n = 2

    result_dict[w[:n]] = i

print(result_dict)