# probe: a module-level generator iterating an object's field (contrast with the method form)
# axes: op=generator-function flow=for
# CLASSIFICATION @ 2026-08-17: RUNS (6)
#
# ⭐ FIXED 2026-08-17, in two steps that are worth keeping separate. First the
# signature: it was computed before the class contracts existed, so the yield
# type was `object`. Then the frame: a `for` loop keeps its position in a cell,
# a cell cannot survive a suspension, and the body fell to the non-suspending
# path which carries one lane per yield -- an int out of a LIST is three. The
# loop is rewritten into an index loop now, which is what the lazy-iterator
# value synthesis already did for the same reason. Pinned by
# tests/golden/cases/generator_over_a_list.py.
# CPython 3.14 expects: 6

from typing import Iterator


class Bag:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs


def each(b: Bag) -> Iterator[int]:
    for x in b.xs:
        yield x


b = Bag([1, 2, 3])
total = 0
for v in each(b):
    total += v
print(total)
