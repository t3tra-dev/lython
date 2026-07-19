# Closures inside generator bodies: nested defs capturing generator locals
# (list, int parameter), called before the first suspension. Closure calls
# AFTER a yield still lose their target evidence in the generator frame —
# a documented generator residual, not covered here.
from typing import Iterator


def gen() -> Iterator[int]:
    xs: list[int] = [1, 2, 3]

    def total() -> int:
        n: int = 0
        for x in xs:
            n += x
        return n

    first: int = total()
    xs[0] = 10
    second: int = total()
    yield first
    yield second
    yield len(xs)


def offsets(base: int) -> Iterator[int]:
    def shifted(x: int) -> int:
        return x + base

    a: int = shifted(1)
    b: int = shifted(2)
    yield a
    yield b


for value in gen():
    print(value)
for value in offsets(100):
    print(value)
