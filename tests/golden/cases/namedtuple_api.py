# What this pins: the four NamedTuple members CPython's namedtuple builds at
# class creation, none of which this compiler had.
#
#     print(P._fields)     # attr.get type object has no static runtime
#                          # attribute '_fields'
#     print(p._replace(x=5))
#     print(p._asdict())   # 'P' inherits builtins.object._replace/_asdict,
#                          # which Lython does not implement
#     print(tuple(p))      # static type !py.contract<"P"> does not provide
#                          # manifest method '__iter__'
#
# Every one of them is decided by the class statement: the field names, their
# order and their types are all known where the class is emitted, and CPython
# builds these out of _fields and _make with nothing dynamic in them. So
# _asdict is a synthesized method returning the dict literal its fields spell,
# _fields folds to a tuple of strings the way C.__name__ folds to a string, and
# _replace and tuple()/list() are rewrites of the call.
#
# Why this must run: `_replace` must not re-evaluate its receiver -- CPython
# calls make() once and so must the rewrite, which fills every field the call
# does not name from that one value. A counter is the only way to see it.
#
# ⛔ The dict from _asdict has the JOIN of the field types as its value type, so
# a mixed NamedTuple gives dict[str, int | str] rather than a refusal -- the
# same type the equivalent literal gets.
#
# ⛔ Only tuple() and list() over a NamedTuple: a set would drop duplicate
# field values and a dict has no keys to use, so neither is the same answer.
# `for v in p` is still refused; the display is a rewrite, not an __iter__.
from typing import NamedTuple


class P(NamedTuple):
    x: int
    y: str = "d"


class Same(NamedTuple):
    a: int
    b: int


calls = 0


def make() -> P:
    global calls
    calls += 1
    return P(2, "m")


p = P(1, "a")
print(p, p.x, p.y, len(p), p[0])
print(P._fields, Same._fields)
print(p._asdict(), Same(1, 2)._asdict())
print(p._replace(x=5), p._replace(y="z"), p._replace(x=9, y="q"), p._replace())
print(make()._replace(x=3), "make calls", calls)
print(tuple(p), list(p), tuple(Same(1, 2)))
print(P(1, "a")._replace(x=2) == P(2, "a"))

n = 0
i = 0
while i < 200:
    row = P(i, "r")
    n += len(row._asdict()) + len(tuple(row)) + row._replace(x=i + 1).x
    i += 1
print("loop", n)
