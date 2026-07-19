# Returned closures: the nonlocal counter pattern (R6 acceptance), a local
# cell escaping its frame, independent instances, and by-value int captures.
from typing import Callable


def counter() -> Callable[[], int]:
    n: int = 0

    def inc() -> int:
        nonlocal n
        n += 1
        return n

    return inc


def make_adder(base: int) -> Callable[[], int]:
    def add() -> int:
        return base + 1

    return add


def snapshot() -> Callable[[], int]:
    n: int = 5

    def get() -> int:
        return n

    return get


f = counter()
print(f())
print(f())
print(f())
g = counter()
print(g())
print(f())

a = make_adder(41)
print(a())

s = snapshot()
print(s())
