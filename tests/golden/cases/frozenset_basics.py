fs = frozenset({3, 1, 2})
print(len(fs))
print(2 in fs, 9 in fs)
fs2 = frozenset([1, 2, 2, 3, 1])
print(len(fs2))
print(fs == fs2, fs != fs2)
u = fs | frozenset([4])
print(len(u))
i = fs & frozenset([2, 3, 9])
print(len(i))
d = fs - frozenset([1])
print(len(d))
x = fs ^ frozenset([3, 4])
print(len(x))
print(fs.issubset(u), u.issuperset(fs), fs.isdisjoint(frozenset([9])))
total = 0
for v in fs:
    total = total + v
print(total)
h1 = hash(frozenset([1, 2, 3]))
h2 = hash(frozenset([3, 2, 1]))
print(h1 == h2)
dd = {frozenset([1, 2]): "a"}
print(dd[frozenset([2, 1])])
e = frozenset()
print(len(e))

# frozenset had no __bool__, so a truth test on one failed at lowering.
print(bool(fs), bool(e))
if fs:
    print("non-empty is true")
if not e:
    print("empty is false")
