print("a,b,,c".split(","))
print("a,b,c".split(",", 1))
print("a,b,c".split(",", 0))
print("  one   two  three ".split())
print("".split())
print("   ".split())
print("aaa".split("aa"))
print("aaa".rsplit("aa"))
print("a,b,c".rsplit(",", 1))
print("one two three".rsplit())
print("ab c\n\nde fg\rkl\r\n".splitlines())
print("ab c\n\nde fg\rkl\r\n".splitlines(True))
print("line1 line2".splitlines())
print("".splitlines())
print("key=value=rest".partition("="))
print("key=value=rest".rpartition("="))
print("plain".partition(":"))
print("plain".rpartition(":"))
print(", ".join(["one", "two", "three"]))
print("-".join("spam".split("a")))
print("".join(["a", "b", "c"]))
print("|".join([]))
print("・".join(["あ", "い"]))
parts = "2026-07-17".split("-")
print(len(parts))
print(parts[0], parts[1], parts[2])
total = 0
for piece in "1 22 333".split():
    total = total + len(piece)
print(total)
try:
    "abc".split("")
except ValueError as e:
    print("split:", e)
try:
    "abc".partition("")
except ValueError as e:
    print("partition:", e)
