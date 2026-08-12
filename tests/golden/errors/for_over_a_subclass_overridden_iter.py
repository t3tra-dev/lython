# `for v in b` calls b's __iter__, and a subclass overrides it: the loop is the
# spelling that used to walk past the guard `b.__iter__()` already had, inline
# the BASE's method and iterate the wrong sequence. Every dunder the emitter
# drives now asks one gate, and the gate asks this question first.
from typing import Iterator


class Base:
    def __init__(self, n: int) -> None:
        self.n = n

    def __iter__(self) -> Iterator[int]:
        return iter([1, 2])


class Sub(Base):
    def __iter__(self) -> Iterator[int]:
        return iter([9, 9, 9])


def walk(b: Base) -> int:
    total = 0
    for v in b:
        total += v
    return total


print(walk(Sub(0)))
