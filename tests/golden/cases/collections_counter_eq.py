from collections import Counter

c = Counter(["a", "a", "b"])
c2 = Counter(["b", "a", "a"])
c3 = Counter(["a", "b"])
print(c == c2, c == c3)
