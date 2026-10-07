# WHAT: a bool passed to an `int | None` parameter -- or an int to a
# `float | None` one -- stays what it is, as CPython leaves the annotation
# inert: it prints True, its repr is True, and it still adds as 1. A second
# union parameter given one of its own members does not get in the way, and a
# default a rung below its union annotation is read the same way.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the call compiles to a
# second body read at the argument's rung, and the claim is about the VALUE
# that body sees -- that `v` prints True and `v + 1` is 2 -- which only running
# it shows (the decode test: a print alone can look right over the wrong
# representation).

def f(v: int | None) -> None:
    print(v, repr(v))
    if v is not None:
        print(v + 1, v * 3)
f(True)
f(False)
f(7)
f(None)
def g(v: float | None, w: int | str) -> str:
    return f"{v} {w}"
print(g(True, False), g(2, "s"), g(None, 3), g(1.5, True))
def h(v: complex | int) -> None:
    print(v)
h(True)
h(2.5)
def k(n: int | None = True) -> None:
    print(n)
k()
