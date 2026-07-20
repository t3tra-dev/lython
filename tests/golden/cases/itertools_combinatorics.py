from itertools import product, combinations, combinations_with_replacement

for t in product([1, 2], "ab"):
    print(t)
for t in combinations("abcd", 2):
    print(t)
for t in combinations_with_replacement([0, 1], 2):
    print(t)
