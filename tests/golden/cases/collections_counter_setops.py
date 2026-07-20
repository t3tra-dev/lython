from collections import Counter

c = Counter(["a", "b", "a", "c", "a", "b"])
d = Counter(["a", "d", "d"])
o = c | d
print(o.most_common())
a = c & d
print(a.most_common())
