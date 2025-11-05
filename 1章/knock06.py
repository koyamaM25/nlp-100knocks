def character_ngram(string, n):
    ngram = []
    for i in range(len(string) - n + 1):
        ngram.append(string[i:i + n])
    return set(ngram)

s1 = 'paraparaparadise'
s2 = 'paragraph'

x = character_ngram(s1, 2)
y = character_ngram(s2, 2)

print(x)
print(y)

union = x | y
intersection = x & y
difference = x - y

print(union)
print(intersection)
print(difference)

if 'se' in x:
    print('xにはseが含まれます')
else:
     print('xにはseが含まれません')

if 'se' in y:
    print('yにはseが含まれます')
else:
     print('yにはseが含まれません') 