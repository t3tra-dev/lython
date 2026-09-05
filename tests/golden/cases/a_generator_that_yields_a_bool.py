# What: a generator whose yielded value is a bool. Its manifest value shape is a
# bare i1, and a suspension lane held (pointer, size) pairs -- so the state
# machine declined every such generator and the tier below refused it for a
# limit it was never there for. Runtime values, because the question is which
# value each resume hands back across the suspension.
from typing import Iterator


def evens(n: int) -> Iterator[bool]:
    for i in range(n):
        yield i % 2 == 0


def literals() -> Iterator[bool]:
    yield True
    yield False
    yield True


def through_call(n: int) -> Iterator[bool]:
    def positive(v: int) -> bool:
        return v > 0

    for i in range(n):
        yield positive(i - 1)


def branched(n: int) -> Iterator[bool]:
    for i in range(n):
        if i % 3 == 0:
            yield True
        else:
            yield i % 2 == 1


class Rows:
    def flags(self, n: int) -> Iterator[bool]:
        for i in range(n):
            yield i < 2


print(list(evens(4)))
print(list(literals()))
print(list(through_call(3)))
print(list(branched(4)))
print(list(Rows().flags(4)))
for flag in literals():
    print(flag)
