import functools
from functools import reduce


def add(a: int, b: int) -> int:
    return a + b


def mul(a: int, b: int) -> int:
    return a * b


def wider(a: str, b: str) -> str:
    return a + "|" + b


# No initial value: the first element seeds the accumulation.
print(reduce(add, [1, 2, 3, 4, 5]))
print(reduce(mul, [1, 2, 3, 4, 5]))
print(reduce(add, [42]))

# With an initial value.
print(reduce(add, [1, 2, 3], 100))
print(reduce(mul, [2, 3], 10))

# A second instantiation of the same generic, over str.
print(reduce(wider, ["a", "b", "c"]))
print(reduce(wider, ["b", "c"], "a"))

# Qualified call reaches the same registration.
print(functools.reduce(add, [7, 8], 5))

# An initial value makes an empty sequence well defined.
empty: list[int] = []
print(reduce(add, empty, 0))
