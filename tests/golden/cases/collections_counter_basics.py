from collections import Counter

c = Counter(["a", "b", "a", "c", "a", "b"])
print(c["a"], c["b"], c["c"], c["z"])
print(c.total())
print(c.most_common())
print(c.most_common(2))
print(c.elements())
