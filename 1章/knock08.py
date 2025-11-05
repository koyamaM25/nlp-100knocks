def cipher(string):
    result = ''
    for ch in string:
        if ch.islower():
            result += chr(219 - ord(ch))
        else:
            result += ch
    return result

message = "Hello, World! 暗号化・復号化"
encrypted = cipher(message)
decrypted = cipher(encrypted)

print("元の文字列 :", message)
print("暗号化文字列 :", encrypted)
print("復号化文字列 :", decrypted)