d = {"a": 1, "b": 2}
k = "a"
print(d[k])
k2 = "zz"
try:
    print(d[k2])
except KeyError as e:
    print("caught:", repr(e))
missing = "gone"
print(d[missing])
