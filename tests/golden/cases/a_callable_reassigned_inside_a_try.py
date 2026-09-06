# What: a `Callable`-typed local reassigned inside a `try`. The rule for what
# can be carried out of the statement read "a contract-typed slot", and a
# function object is not a contract -- so the retry idiom every fallback is
# written as was refused outright, while the same local reassigned inside a
# `for`, a `with` or a `match` takes that very cell.
#
# ⭐ Third reader of one stale enumeration: the module globals and the region
# slot rule each excluded a callable for the same unstated reason, and each was
# a program that worked one spelling over.
#
# Running it is the evidence the reassignment reached the continuation: which
# body runs is what the printed number says, and the two flags take different
# paths through the same statement.
from typing import Callable


def double(n: int) -> int:
    return n * 2


def triple(n: int) -> int:
    return n * 3


def boom(flag: bool) -> int:
    if flag:
        raise ValueError("x")
    return 0


def by_name(flag: bool) -> int:
    op: Callable[[int], int] = double
    try:
        boom(flag)
        op = triple
    except ValueError:
        op = double
    return op(4)


def by_lambda(flag: bool) -> int:
    op: Callable[[int], int] = lambda n: n
    try:
        boom(flag)
    except ValueError:
        op = lambda n: n + 100
    return op(4)


def through_finally(flag: bool) -> int:
    op: Callable[[int], int] = lambda n: n
    try:
        boom(flag)
        op = lambda n: n + 1
    except ValueError:
        pass
    finally:
        pass
    return op(4)


print("by name", by_name(False), by_name(True))
print("by lambda", by_lambda(False), by_lambda(True))
print("through finally", through_finally(False), through_finally(True))
