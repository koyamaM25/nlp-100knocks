def character_ngram(string, n):
    ngram = []
    for i in range(len(string) - n + 1):
        ngram.append(string[i:i + n])
    return ngram

def word_ngram(string, n):
    string =string.split(' ')
    ngram = []
    for i in range(len(string) - n + 1):
        ngram.append(string[i:i + n])
    return ngram

s = 'I am an NLPer'

print(character_ngram(s, 1))
print(character_ngram(s, 2))
print(character_ngram(s, 3))

print(word_ngram(s, 1))
print(word_ngram(s, 2))
print(word_ngram(s, 3))