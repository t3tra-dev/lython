table = str.maketrans("abc", "xyz")
print(len(table))
print("cabbage".translate(table))
up = {ord("l"): "L", ord("o"): "0"}
print("hello world".translate(up))
print("no-op".translate({}))
try:
    str.maketrans("ab", "xyz")
except ValueError as e:
    print("ValueError:", e)
