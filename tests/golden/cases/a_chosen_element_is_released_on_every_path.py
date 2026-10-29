# Why execution: the leak gate measures this program (tests/CMakeLists.txt),
# and only a run can count what each call leaves behind.
#
# Three choices of one object among several, each of which kept one too many
# or one too few references on a path the others did not take:
# - a display element read at a run-time index in one arm of a ternary: the
#   read retains the element it chose, and the element's own reference was
#   dropped from the arm that reads (one Cell per call);
# - an IndexError from a run-time index into a display, caught in the same
#   function: the elements kept for the read had no release on the edge to
#   the handler (two lists per call);
# - a generator keeping `a if f else b` across a `yield`: the suspend took a
#   reference for the frame and so did the borrowed-return rule (one object
#   per generator).
# - `type(o)` of an element of a display walked in a loop inside a loop: the
#   type object read from the element's header was counted as a view of the
#   element, and handing it to a merge dropped the element's release (one
#   object per trip of the outer loop).
from typing import Iterator


class Cell:
    def __init__(self, text: str) -> None:
        self.text = text


def first(k: int) -> str:
    row = (Cell("a"), Cell("b"))
    w = row[k] if k < 1 else row[0]
    return w.text


def nth(i: int) -> int:
    t = ([1], [2])
    try:
        return len(t[i])
    except IndexError:
        return -1


def gen(a: Cell, b: Cell, f: bool) -> Iterator[str]:
    x = a if f else b
    yield x.text
    yield x.text + "!"


class Leaf(Cell):
    pass


def kinds(trips: int) -> list[str]:
    seen: list[str] = []
    for _ in range(trips):
        for o in (Cell("c"), Leaf("l")):
            seen.append(type(o).__name__)
    return seen


out: list[str] = []
total = 0
for i in range(100):
    out.append(first(i % 2))
    total += nth(i % 3)
    out.extend(gen(Cell("p"), Cell("q"), i % 2 == 0))
print(total, len(out), out[:6])
names = kinds(100)
print(len(names), names[:4])
