# \xNN and octal escapes denote CODEPOINTS in str literals (landing as
# their UTF-8 encoding) but raw BYTES in bytes literals.
s = "\xc3\xa9"
print(s)
print(len(s))
print("\xe9" == "é")
print("\x41\x7f" == "A\x7f")
print(len("\x80"))
print("\101\102\77")
print(len("\377"))
b = b"\xc3\xa9\377\101"
print(b)
print(len(b))
print("caf\xe9".encode())
print("caf\xe9".encode().decode() == "café")
print("é" == "\xe9")
print(len("\U0001F600"))
