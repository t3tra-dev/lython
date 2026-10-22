# WHAT: a class -- the program's, and builtin `int` -- given to a Callable
# parameter (the program's to a generic one, whose type arguments its
# constructor answers), stored in a Callable-typed name and list, returned
# from a function, and mapped: each call constructs an instance.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: a class is a value with no
# lanes, and what makes it callable is a forwarder `return C(a0, ...)` the
# emitter builds for the callable it is handed to; that the right constructor
# ran is seen in the instances it made.
from collections.abc import Callable


class Point:
    def __init__(self, x: int, y: int = 0) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return "Point(%d, %d)" % (self.x, self.y)


def apply1[A, R](f: Callable[[A], R], value: A) -> R:
    return f(value)


def parse(f: Callable[[str], int], text: str) -> int:
    return f(text) + 1


def factory() -> Callable[[int], Point]:
    return Point


print(apply1(Point, 3))
print(parse(int, "41"))
print(parse(len, "abc"))
make: Callable[[int, int], Point] = Point
print(make(1, 2))
print(factory()(7))
parsers: list[Callable[[str], int]] = [int, len]
print([p("12") for p in parsers])
print(list(map(Point, [4, 5])))
