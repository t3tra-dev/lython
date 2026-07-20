from collections import Counter

c = Counter(["a", "b", "a", "c", "a", "b"])
d = Counter(["a", "d", "d"])
s = c + d
print(s.most_common())
u = c - d
print(u.most_common())
