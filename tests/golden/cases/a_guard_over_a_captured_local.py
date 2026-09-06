# What: a guard over a local that a nested callable reads and the body REBINDS.
# Such a name lives in a cell, and the narrowing unwrapped `values[name]` --
# which for a cell is the cell OBJECT and never a union -- so the proof was
# silently dropped and the read handed the whole union on:
#
#     def make() -> int:
#         v: Optional[int] = 3
#         def inner() -> int:
#             if v is None: return 0
#             return v          # cannot adapt union<int, None> return value
#         v = 4
#         return inner()
#
# while `w = v` inside `inner` compiled and printed CPython's answer -- the same
# workaround the union FIELD family had, and the same repair: a cell is RE-READ
# like a field, so the proof is spent at the read, with a check.
#
# Running it is the whole evidence: the rebinding before the call is what the
# closure must see, and the guarded arm has to return the payload rather than
# the union. The `is None` path is exercised too, because the check the
# narrowing installs must not fire where the guard did not prove anything.
from typing import Optional


class Shape:
    pass


class Named(Shape):
    def tag(self) -> str:
        return "named"


def counted(seed: Optional[int], replace: bool) -> int:
    v = seed

    def inner() -> int:
        if v is None:
            return 0
        return v + 1

    if replace:
        v = 41
    return inner()


def ternary(seed: Optional[str]) -> int:
    v = seed

    def inner() -> int:
        return 0 if v is None else len(v)

    v = "abcd"
    return inner()


def classes(seed: Shape) -> str:
    v = seed

    def inner() -> str:
        if isinstance(v, Named):
            return v.tag()
        return "plain"

    v = Named()
    return inner()


print("rebound", counted(3, True), counted(3, False))
print("none stays none", counted(None, False))
print("ternary", ternary(None), ternary("z"))
print("class guard", classes(Shape()), classes(Named()))
