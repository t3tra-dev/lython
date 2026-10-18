# WHAT: a traceback names each frame as CPython does -- a nested def by its
# own name, a lambda as `<lambda>`, a method by its name -- whatever symbol
# the function is emitted under.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the frames are recorded
# as the exception unwinds at run time. A nested def's frame printed its
# internal symbol (`outer$inner$2$15_4`), and so did a lambda's.
import traceback
from typing import Callable


def frames(e: BaseException) -> list[str]:
    out: list[str] = []
    for line in traceback.format_exception(e):
        first = line.rstrip().split("\n")[0]
        if first.startswith("  File"):
            out.append(first.split(", in ")[1])
    return out


def outer(k: int) -> int:
    def inner(n: int) -> int:
        def innermost(m: int) -> int:
            return 10 // m
        return innermost(n)
    return inner(k)


class Box:
    def run(self, f: Callable[[int], int], x: int) -> int:
        return f(x)


def via_lambda(x: int) -> int:
    g: Callable[[int], int] = lambda v: 100 // v
    return Box().run(g, x)


for test in [lambda: outer(0), lambda: via_lambda(0)]:
    try:
        test()
    except ZeroDivisionError as e:
        print(frames(e))
