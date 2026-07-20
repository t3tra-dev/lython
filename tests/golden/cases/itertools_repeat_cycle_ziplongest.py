from itertools import repeat, cycle, zip_longest, chain, islice

for s in repeat("xy", 3):
    print(s)
n = 0
for c in cycle("ab"):
    print(c)
    n = n + 1
    if n >= 5:
        break
for a, b in zip_longest([1, 2, 3], [10], fillvalue=0):
    print(a, b)
for v in chain.from_iterable([[1, 2], [3, 4]]):
    print(v)
r = 0
for v in islice(repeat(7), 3):
    r = r + v
print(r)
