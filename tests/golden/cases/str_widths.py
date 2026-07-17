# Adaptive-width (PEP 393 style) str semantics: len and indexing count code
# points across latin-1 / UCS-2 / UCS-4 payloads, concatenation and
# comparison mix widths, and encode() round-trips through UTF-8.
s2 = "caf\xe9"
print(len(s2))
print(s2[3])
print(s2[-1] == "\xe9")
s3 = "Āあz"
print(len(s3))
print(s3[1])
s4 = "a\U0001F600b"
print(len(s4))
print(s4[1])
print(s4[-2] == "\U0001F600")
mix = s2 + s3
print(len(mix))
print(mix[4])
print(mix == "caf\xe9Āあz")
print("Ā" < "ā")
print("z" < "Ā")
print("\xe9" < "Ā")
for ch in "\xe9Ā\U0001F600":
    print(ch)
print("a\U0001F600b".encode())
print("あい".encode().decode() == "あい")
empty = ""
print(len(empty))
print(empty + "x")
