# probe: a module-level generator iterating an object's field (contrast with the method form)
# axes: op=generator-function flow=for
# CLASSIFICATION @ 2026-08-17: 3 loud 拒否 (診断)
#   source generator next lowering currently supports yields whose runtime
#   value is a single lane, and '!py.contract<"builtins.int">' has 3
#
# ⛔ RECLASSIFIED. It used to be "generator function return annotation is
# incompatible with inferred Generator or AsyncGenerator contract", which was
# the SIGNATURE being computed before the class contracts existed -- fixed
# 2026-08-17, and the probe moved one layer down to the real blocker. The yield
# type is now `int`; what stops it is that an int read out of a LIST is the
# 3-lane object form, while the generator frame carries a single lane. That is
# the int-only yield plan recorded on the seven-gap cluster, and
# `for i in range(n): yield i` works because a range element rides an i64 lane.
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
