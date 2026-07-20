from itertools import pairwise, accumulate

for a, b in pairwise([1, 2, 3, 4]):
    print(a, b)
for a, b in pairwise("abc"):
    print(a, b)
for v in accumulate([1, 2, 3, 4]):
    print(v)
for v in accumulate([5, 2, 8], lambda a, b: a * b):
    print(v)
