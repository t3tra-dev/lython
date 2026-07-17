b = b"the quick brown fox"
print(b.find(b"quick"))
print(b.find(b"cat"))
print(b.count(b"o"))
print(b.count(b""))
print(b.startswith(b"the"))
print(b.startswith(b"quick", 4))
print(b.endswith(b"fox"))
print(b"  spam  ".strip())
print(b"xxspamxx".strip(b"x"))
print(b"a,b,,c".split(b","))
print(b" one  two ".split())
print(b"a,b,c".split(b",", 1))
print(b"one one".replace(b"one", b"two"))
print(b"one one".replace(b"one", b"two", 1))
print(b"deadBEEF".hex())
print(b"\x00\xff\x10".hex())
print(bytes.fromhex("deadbeef"))
print(bytes.fromhex("2Ef0 F1f2  "))
print(bytes.fromhex(""))
print(b", ".join([b"a", b"bb", b"ccc"]))
print(b"|".join([]))
print(b"ab" * 3)
print(b"x" * 0)
print(b"hello".decode())
print(b"hello".decode("utf-8"))
print(b"h\xc3\xa9llo".decode("UTF-8", "strict"))
for chunk in b"a b c".split():
    print(chunk)
try:
    bytes.fromhex("abx")
except ValueError as e:
    print("fromhex:", e)
try:
    bytes.fromhex("abc")
except ValueError as e:
    print("odd:", e)
try:
    b"a,b".split(b"")
except ValueError as e:
    print("split:", e)
