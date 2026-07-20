from itertools import chain, takewhile, dropwhile, filterfalse

for v in chain([1, 2], [7, 8], [9]):
    print(v)
g = chain("ab", "cd")
print(next(g))
for s in g:
    print(s)
xs = [1, 4, 6, 4, 1]
for v in takewhile(lambda x: x < 5, xs):
    print("t", v)
for v in dropwhile(lambda x: x < 5, xs):
    print("d", v)
for v in filterfalse(lambda x: x % 2 == 0, [1, 2, 3, 4, 5]):
    print("f", v)
