# Counter's arithmetic, ordering and views over a str-keyed count.
#
# One line does NOT match CPython, and it is a recorded port deviation:
# `elements()` returns a materialized list where CPython returns a lazy
# itertools.chain, so `print(c.elements())` renders the elements instead of
# an object repr (collections.py documents it; CPython's own output here is
# an address, which no golden could pin anyway). Noted because a differential
# run over the golden corpus reads this file's stdout as a divergence.
from collections import Counter

c = Counter(["a", "b", "a", "c", "a", "b"])
print(c["a"], c["b"], c["c"], c["z"])
print(c.total())
print(c.most_common())
print(c.most_common(2))
print(c.elements())
