n = 7
d = {"a": 1}
d["b"] = 2
print(min(n, len(d)))
print(max(n, len(d)))
print(min(3, 5))
print(max("a", "b"))
print(min(2.5, 1.5))
a = 1
b = 2
if a < 2 and b > 1:
    print("both")
if a > 5 or b > 1:
    print("either")
print(a < 2 and b > 5)
print(a < 2 and b > 1 and n > 2)
if "y" in d and d["y"] > 0:
    print("unreachable")
print("n =", n, "len =", len(d))
print("xs:", [1, 2], 2.5, True)
