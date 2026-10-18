# WHAT: a call through a callable value whose declared result is `object` or
# a union answers with what the function it holds returns -- an int, a str,
# None, a bool, a float, a union, a tuple, a dict, an instance of a subclass
# of the class the function declares -- through a parameter with one
# candidate, with three (arms at the call), and with eight (a dispatcher);
# the zero-argument lambdas CPython's tests hand to assertRaises; and a union
# with None seen as `object` without any callable.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the result is laid out as
# the target returns it and seen as the declared type in the lowering, and
# which class the box names (B's repr through `mk() -> A`) is a run-time
# answer. Every call here was refused by the lowering, which read the
# target's lanes as the declared type's.
from typing import Callable


class A:
    def __repr__(self) -> str:
        return "A()"


class B(A):
    def __repr__(self) -> str:
        return "B()"


def mk() -> A:
    return B()


def one() -> int:
    return 1


def none() -> None:
    return None


def flag() -> bool:
    return True


def half() -> float:
    return 0.5


def maybe(n: int) -> int | None:
    return n if n > 0 else None


def show(f: Callable[[], object]) -> None:
    print(f())


show(one)


def run(f: Callable[[], object]) -> None:
    v = f()
    print(repr(v), str(v), v == v, v is None)


gs: list[Callable[[], object]] = [
    mk, none, flag, half,
    lambda: maybe(3), lambda: maybe(0), lambda: (1, "a"), lambda: {"k": [1]},
]
for g in gs:
    run(g)


def opt(f: Callable[[], int | None]) -> int:
    v = f()
    return -1 if v is None else v * 2


print(opt(lambda: None), opt(lambda: 21), opt(lambda: maybe(4)),
      opt(lambda: maybe(-4)))


def raises(exc: type[BaseException], f: Callable[[], object]) -> bool:
    try:
        f()
    except BaseException as e:
        return isinstance(e, exc)
    return False


print(raises(ZeroDivisionError, lambda: 1 // 0),
      raises(KeyError, lambda: {"a": 1}["x"]),
      raises(IndexError, lambda: [1][3]))

x: object = maybe(5)
y: object = maybe(-5)
print(x, y, x is None, y is None)
