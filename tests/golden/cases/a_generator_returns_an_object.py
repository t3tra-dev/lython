# A generator's return value of a type other than int: read by a delegating
# `yield from`, dropped by a `for` loop, and carried by the StopIteration that
# next() raises. The returned object crosses the resume as a value of its own
# type, and the printed lines are where it arrives.
from typing import Generator


class Box:
    def __init__(self, v: int) -> None:
        self.v = v


def named(n: int) -> Generator[int, None, str]:
    for i in range(n):
        yield i
    return "done" + str(n)


def boxed(n: int) -> Generator[int, None, Box]:
    yield n
    return Box(n * 10)


def outer() -> Generator[int, None, None]:
    r = yield from named(2)
    print("got", r)
    b = yield from boxed(3)
    print("box", b.v)


for v in outer():
    print(v)
for v in named(1):
    print(v)
g = named(1)
print(next(g))
try:
    next(g)
except StopIteration as e:
    print("stop:", str(e))
