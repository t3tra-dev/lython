# probe: REPORTED loud: a generator method on a class
# axes: op=generator-method flow=for
# CLASSIFICATION @ kernel/4a 6c328b5: 3 loud 拒否 (診断)
#   generator function return annotation is incompatible with inferred Generator or AsyncGenerator contract
# CPython 3.14 expects: 6

from typing import Iterator


class Bag:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs

    def each(self) -> Iterator[int]:
        for x in self.xs:
            yield x


b = Bag([1, 2, 3])
total = 0
for v in b.each():
    total += v
print(total)
