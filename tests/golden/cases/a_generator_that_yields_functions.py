# What: a generator whose yield type is a `Callable`. An unannotated lambda has
# no type of its own -- its parameters read as `object` -- so the whole-body
# walk reported the function as a mismatch ("annotated Iterator[Callable[[int],
# int]] but yields Callable[[object], ...]") and, once that was taken, the
# lambda itself had nothing to read its parameters against ("lambda requires a
# Callable annotation because its type contains unresolved Unknown"). Both
# halves are the same fact: the DECLARED yield type is the expectation, exactly
# as it is for `v: Callable[[int], int] = lambda n: n + 1` one line over.
#
# Running it is what shows which function came out: each yielded lambda closes
# over a different number, so the printed values name them.
from typing import Callable, Iterator


def steps() -> Iterator[Callable[[int], int]]:
    yield lambda n: n + 1
    yield lambda n: n * 10


def named() -> Iterator[Callable[[int], int]]:
    def twice(n: int) -> int:
        return n * 2

    yield twice
    yield lambda n: n - 1


lambdas: list[int] = []
for op in steps():
    lambdas.append(op(4))
print("lambdas", lambdas)

mixed: list[int] = []
for op in named():
    mixed.append(op(4))
print("mixed", mixed)
