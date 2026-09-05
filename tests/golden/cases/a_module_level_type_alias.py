# What: a name bound to a TYPE and then spelled in annotations. Every form --
# a builtin, a declared class, a generic subscript, a `|` union, PEP 695's
# `type X = ...`, an alias of an alias, and an alias inside a container
# annotation. Runtime values, because the question is which type each annotated
# name resolves to and what the annotated code then does with it; the
# alternative answer was a contract invented out of the spelling.
from typing import Callable, Optional

Name = str
Row = list[int]
Maybe = Optional[int]
Num = int | float
Chain = Maybe
type Tag = str


class Widget:
    def __init__(self, n: int) -> None:
        self.n: int = n


W = Widget
Table = dict[Name, int]


def shout(v: Name) -> Name:
    return v + "!"


def total(r: Row) -> int:
    return sum(r)


def describe(v: Maybe, c: Chain) -> str:
    return str(v) + "/" + str(c)


def widen(v: Num) -> str:
    return str(v)


def label(t: Tag) -> Tag:
    return t + "?"


def unwrap(w: W) -> int:
    return w.n


def lookup(table: Table, key: Name) -> int:
    return table[key]


def apply(f: Callable[[Name], Name], v: Name) -> Name:
    return f(v)


print(shout("a"))
print(total([1, 2, 3]))
print(describe(1, None))
print(widen(1), widen(1.5))
print(label("t"))
print(unwrap(Widget(4)))
print(lookup({"a": 1}, "a"))
print(apply(shout, "b"))
