# Helper for an_imported_decorator_is_applied. Two decorators declared beside
# the functions they wrap, plus a driver inside the module so the decorated
# name has to answer the wrapper on BOTH sides of the boundary.
from typing import Callable


def times_ten(f: Callable[[int], int]) -> Callable[[int], int]:
    def wrapped(n: int) -> int:
        return f(n) * 10

    return wrapped


def plus_one(f: Callable[[int], int]) -> Callable[[int], int]:
    def wrapped(n: int) -> int:
        return f(n) + 1

    return wrapped


@times_ten
def scaled(n: int) -> int:
    return n + 1


@times_ten
@plus_one
def both(n: int) -> int:
    return n


def driver(n: int) -> int:
    return scaled(n)
