# WHAT: a call through a `Callable[[int], int]` value whose target only the
# function object knows, with more candidates of that type than the call site
# writes out -- so it goes through the shared dispatcher the lowering makes for
# that call shape. Each candidate returns, raises into a `try` around the call,
# and finally raises out of the program; the traceback names the caller and the
# callee and nothing between them.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the dispatcher is a function
# of the compiled program with the call's own location, and what it must not
# do -- add a traceback frame, or lose the one its caller pushes -- shows only
# in what the running program prints. So does the value an arm hands back: a
# big int, an owned str, a closure's captured cell.
#
# ⛔ FOUR PLAIN FUNCTIONS AND FOUR CLOSURES, in separate lists: below four
# candidates the dispatch stays at the call, and a list mixing the two is
# refused today for a reason unrelated to this case.
from typing import Callable


def inc(x: int) -> int:
    return x + 1


def dbl(x: int) -> int:
    return x * 2


def picky(x: int) -> int:
    if x < 0:
        raise ValueError("negative: " + str(x))
    return x


def big(x: int) -> int:
    return x * 1000000000000000000000


def adder(n: int) -> Callable[[int], int]:
    def add(x: int) -> int:
        return x + n
    return add


def apply(f: Callable[[int], int], x: int) -> int:
    return f(x)


def guarded(f: Callable[[int], int], x: int) -> int:
    try:
        return f(x)
    except ValueError as error:
        print("caught", error)
        return -1


fs = [inc, dbl, picky, big]
for f in fs:
    print(apply(f, 3), guarded(f, -2))
for h in [adder(10), adder(20), adder(30), adder(40)]:
    print(apply(h, 3), guarded(h, -2))


def maybe(x: int) -> int | None:
    return None if x == 0 else x


def name(x: int) -> str:
    return "n" + str(x)


def pick(x: int) -> str:
    return "p" * x


def last(x: int) -> str:
    return str(x)[-1]


def call_s(f: Callable[[int], str], x: int) -> str:
    return f(x)


def lam(x: int) -> str:
    return "lam" + str(x)


for g in [name, pick, last, lam]:
    print(call_s(g, 3))
print(apply(picky, -5))
