import random

def typoglycemia(input_word):
    len_input = len(input_word)
    want_shuffle = input_word[1:len_input-1]
    want_shuffle = list(want_shuffle)
    random.shuffle(want_shuffle)
    shuffled_word = str(''.join(want_shuffle))

    return input_word[0] + shuffled_word +input_word[len_input-1]

s = 'I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'

words = s.split(' ')
get_typoglycemia_str = ''

for ch in words:
    if len(ch) > 4:
        ch = typoglycemia(ch)
    get_typoglycemia_str += ch + ' '

get_typoglycemia_str = get_typoglycemia_str[:-1] 
print(get_typoglycemia_str)