# What this pins: `Counter.keys()/values()/items()`, and a generic class's own
# method taking its receiver spelled WITHOUT type arguments.
#
# Two defects met in one expression, `sorted((c + c).items())`:
#
#   `collections.py`'s docstring lists keys()/values()/items() under Counter's
#   deviations ("return LISTS, not views"), and Counter did not have them at
#   all -- only OrderedDict did. A documented deviation is only documented if
#   the thing it describes exists.
#
#   `def __add__(self, other: "Counter")` spells the receiver bare, and the
#   argument arrives as `Counter[str]`. The declared-parameter check added
#   with `RejectsMethodArgumentThatViolatesTheDeclaredParameter` refused that:
#   "argument 'other' of '__add__' is declared Counter and this call gives it
#   Counter[str]". A bare generic contract accepts any instantiation of
#   itself; the reverse -- declared WITH arguments, supplied without -- is a
#   real mismatch and stays refused.
#
# Why this needs to run rather than assert on a diagnostic: the views have to
# agree with the counts they came from, and `c + c` has to add rather than
# alias. `items()` returning the keys twice, or `+` returning the receiver,
# both compile.
#
# Every expected line is python3.14's.

from collections import Counter

c: Counter[str] = Counter()
c.update(["a", "b", "a", "c", "a"])

print(sorted(c.keys()))
print(sorted(c.values()))
print(sorted(c.items()))
print(c.total(), len(c), c["a"], c["zz"])


# --- the bare-generic parameter, through +, - and the multiset ops --------
d: Counter[str] = Counter()
d.update(["a", "z"])

print(sorted((c + d).items()))
print(sorted((c - d).items()))
print(sorted((c | d).items()))
print(sorted((c & d).items()))


# --- the views of a result, which is where the two met --------------------
print(sorted((c + c).items()), (c + c).total())
print(sorted((c + d).keys()), sorted((c + d).values()))


# --- OrderedDict's views still answer, since they are the ones that existed
from collections import OrderedDict

o: OrderedDict[str, int] = OrderedDict[str, int]()
o["x"] = 1
o["y"] = 2
# Wrapped in list(): this port returns materialized lists where CPython
# returns odict_* views, which is a recorded deviation and not what this case
# is about (collections.py's docstring).
print(list(o.keys()), list(o.values()), list(o.items()))
