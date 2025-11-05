def make_string(x, y, z):
    return x + '時の' + y + 'は' + z

x = input("xの文字を入力してください: ")
y = input("yの文字を入力してください: ")
z = input("zの文字を入力してください: ")

result = make_string(x,y,z)
print(result)