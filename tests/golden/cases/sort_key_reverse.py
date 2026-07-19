xs = [3, 1, 2]
print(sorted(xs, reverse=True))
print(sorted(xs, key=lambda v: -v))
words = ["bb", "a", "ccc", "dd"]
print(sorted(words, key=len))
print(sorted(words, key=len, reverse=True))
pairs = [("b", 2), ("a", 1), ("b", 1), ("a", 2)]
print(sorted(pairs, key=lambda p: p[0]))
print(sorted(pairs, key=lambda p: p[0], reverse=True))
ys = [5, 3, 9, 1]
ys.sort(reverse=True)
print(ys)
ys.sort(key=lambda v: v % 3)
print(ys)
zs = ["b", "aa", "c"]
zs.sort(key=len, reverse=True)
print(zs)
print(sorted([2, 1], reverse=False))

# stability: equal keys keep source order (forward and reversed)
records = [("x", 1), ("y", 2), ("x", 3), ("y", 4), ("x", 5)]
print(sorted(records, key=lambda r: r[0]))
print(sorted(records, key=lambda r: r[0], reverse=True))
rs = records.copy()
rs.sort(key=lambda r: r[0])
print(rs)

# list.clear / list.extend natives
cs = [1, 2, 3]
cs.clear()
print(len(cs))
cs.extend([7, 8])
cs.extend([9])
print(cs)
