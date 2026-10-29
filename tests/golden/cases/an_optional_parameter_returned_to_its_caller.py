# Why execution: the leak gate measures this program (tests/CMakeLists.txt),
# and only a run can show a reference count going below zero.
#
# A function returning its `Node | None` parameter declared no owned result
# (the program's classes had no deallocator yet when its ABI was laid out),
# so it took no reference at the return while its caller released the one it
# believed it was given: `Ly_DecRef observed non-positive refcount`. The same
# result typed through a dispatcher was owned, and `str | None` always was.
from typing import Callable


class Node:
    def __init__(self, n: int) -> None:
        self.n = n


def ident(e: Node | None) -> Node | None:
    return e


def widen(e: Node) -> Node | None:
    return e


def first_some(xs: list[Node | None], fallback: Node | None) -> Node | None:
    for x in xs:
        if x is not None:
            return x
    return fallback


def coalesce(a: Node | None, b: Node | None) -> Node | None:
    if a is not None:
        return a
    return b


def run() -> int:
    total = 0
    keep = Node(7)
    through: Callable[[Node | None], Node | None] = lambda e: e
    for i in range(100):
        r = ident(Node(i))
        if r is not None:
            total += r.n
        w = widen(Node(1))
        if w is not None:
            total += w.n
        f = first_some([None, None], keep)
        if f is not None:
            total += f.n
        c = coalesce(None, keep)
        if c is not None:
            total += c.n
        t = through(keep)
        if t is not None:
            total += t.n
    return total


print(run())
r = ident(Node(3))
print(type(r).__name__)
if r is not None:
    print(r.n)
print(ident(None))
