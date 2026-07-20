# Wave 2 cross-track pin: foundation's list/tuple/set constructors and
# sorted(<iterable>) consume the generators itercol's itertools desugars
# synthesize in value position. The iterator must be bound to a name first:
# nesting the itertools call directly inside the constructor is rejected at
# emit time (inference sees typing.Any for the nested call), which is the
# documented loud boundary, not part of this pin.
from itertools import chain, islice, repeat, combinations, product

g = chain([3, 1], [2])
print(sorted(g))
g2 = chain([1, 2], [3])
print(list(g2))
g3 = chain([1, 2], [3])
print(tuple(g3))
h = islice("abcdef", 1, 5, 2)
print(list(h))
r = repeat(7, 3)
print(list(r))
c = combinations([1, 2, 3], 2)
print(list(c))
p = product([1, 2], [3, 4])
print(list(p))
