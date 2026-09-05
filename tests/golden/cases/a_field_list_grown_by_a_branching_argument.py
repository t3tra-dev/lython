# What: `self.items.append(<argument that branches>)`. A ternary lowers to a cf
# diamond and a call through a callable VALUE splits the block, so the field is
# read BEFORE those blocks and the append lands in the join -- which a same-block
# test read as a branch the program did not write. Runtime values, because the
# question is what the list holds afterwards; the same append of a literal, of a
# direct call's result, or of a ternary bound to a local first has always worked.
from typing import Callable


class Bag:
    def __init__(self) -> None:
        self.items: list[int] = []
        self.names: list[str] = []

    def add_ternary(self, flag: bool) -> None:
        self.items.append(1 if flag else 2)

    def add_reduction(self, xs: list[int]) -> None:
        self.items.append(sum(v * 2 for v in xs))

    def add_dispatched(self, f: Callable[[int], str], v: int) -> None:
        self.names.append(f(v))

    def add_nested(self, flag: bool, other: bool) -> None:
        self.items.append((10 if other else 20) if flag else 30)


def shout(v: int) -> str:
    return "s" + str(v)


def whisper(v: int) -> str:
    return "w" + str(v)


b = Bag()
b.add_ternary(True)
b.add_ternary(False)
b.add_reduction([1, 2, 3])
b.add_nested(True, True)
b.add_nested(True, False)
b.add_nested(False, True)
b.add_dispatched(shout, 1)
b.add_dispatched(whisper, 2)
print(b.items)
print(b.names)
