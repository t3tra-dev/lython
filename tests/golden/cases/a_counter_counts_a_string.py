# `Counter("mississippi")` -- the example in CPython's own Counter docstring --
# was refused: this port declared `iterable: list[str]`, and typeshed says
# `Iterable[_T]`, of which a str is one. The port's docstring lists its
# deviations from CPython, and this was not one of them; it was an oversight in
# the annotation.
#
# Golden because the repair is a NARROWING, not a signature: iterating a union
# is refused, so `update` branches on isinstance and each arm iterates one
# concrete type. A wrong branch counts the wrong thing and still prints a
# Counter, which only the values catch.
from collections import Counter

c = Counter("mississippi")
print(sorted(c.items()))
print(c.most_common(2))

d = Counter(["a", "b", "a"])
print(sorted(d.items()))
d.update("abc")
print(sorted(d.items()))
d.update(["a"])
print(sorted(d.items()))
d.subtract("ab")
print(sorted(d.items()))
d.subtract(["c"])
print(sorted(d.items()))

print(len(Counter("")), sorted(Counter().items()), Counter([]).most_common(1))
print(sorted((Counter("aab") + Counter("bc")).items()))
