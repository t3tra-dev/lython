from collections import Counter

e = Counter()
e.subtract(["x"])
e.update(["x", "y"])
print(e["x"], e["y"])
print(e.total(), len(e), "x" in e)
