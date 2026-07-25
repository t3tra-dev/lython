# probe: a module-level generator iterating an object's field (contrast with the method form)
# axes: op=generator-function flow=for
# CLASSIFICATION @ kernel/4a 6c328b5: 3 loud 拒否 (診断)
#   generator function return annotation is incompatible with inferred Generator or AsyncGenerator contract
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
