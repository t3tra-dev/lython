# What: a parameter default that stands a rung below its annotation keeps its
# own type, exactly as passing that value would. Runtime values, because the
# whole question is which type the call answers with: `go()` on
# `def go(v: float = 1)` is CPython's 1, not 1.0, and the same function called
# with a float still answers as a float.


def scale(v: float = 1) -> float:
    return v * 2


def offset(v: float = 0) -> float:
    return v + 1.5


def flagged(n: int = True) -> int:
    return n


def mixed(a: float = 1, b: int = 2) -> float:
    return a + b


def named(first: int, second: float = 3) -> float:
    return first + second


print(scale(), scale(2.5), scale(3))
print(offset(), offset(2.5))
print(flagged(), flagged(7))
print(mixed(), mixed(1.5), mixed(1.5, 4))
print(named(1), named(1, 2.5), named(1, 2))
