# WHAT: the StopIteration that next() and send() raise when a generator
# returns carries the returned value itself -- a float, a list, an int, a
# bool, a class with a __repr__, an Optional that holds a value and one that
# holds None -- as repr, str and args show it.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the exception is built at
# run time from the value the generator hands back, and what was wrong was
# its contents: the value went in as str(value), so repr read
# `StopIteration('2.5')` and args[0] was a str, while str() agreed.
from typing import Generator, Optional


class Box:
    def __init__(self, v: int) -> None:
        self.v = v

    def __repr__(self) -> str:
        return "Box(" + str(self.v) + ")"


def gf() -> Generator[int, None, float]:
    yield 1
    return 2.5


def gl() -> Generator[int, None, list[int]]:
    yield 1
    return [1, 2]


def gi() -> Generator[int, None, int]:
    yield 1
    return 7


def gbool(n: int) -> Generator[int, None, bool]:
    yield n
    return n > 2


def gb(n: int) -> Generator[int, None, Box]:
    yield n
    return Box(n)


def gs(n: int) -> Generator[int, int, str]:
    got = yield n
    return "s" * got


def go(flag: bool) -> Generator[int, None, Optional[float]]:
    yield 1
    if flag:
        return 2.5
    return None


def show_float(g: Generator[int, None, float]) -> None:
    next(g)
    try:
        next(g)
    except StopIteration as e:
        print(repr(e), e.args, str(e))


def show_list(g: Generator[int, None, list[int]]) -> None:
    next(g)
    try:
        next(g)
    except StopIteration as e:
        print(repr(e), e.args, str(e))


def show_int(g: Generator[int, None, int]) -> None:
    next(g)
    try:
        next(g)
    except StopIteration as e:
        print(repr(e), e.args, str(e))


def show_optional(g: Generator[int, None, Optional[float]]) -> None:
    next(g)
    try:
        next(g)
    except StopIteration as e:
        print(repr(e), e.args, str(e))


show_float(gf())
show_list(gl())
show_int(gi())
show_optional(go(True))
show_optional(go(False))
g1 = gbool(5)
next(g1)
try:
    next(g1)
except StopIteration as e:
    print(repr(e), e.args)
g2 = gb(3)
next(g2)
try:
    next(g2)
except StopIteration as e:
    print(repr(e), str(e))
g3 = gs(1)
next(g3)
try:
    g3.send(3)
except StopIteration as e:
    print(repr(e), e.args)
