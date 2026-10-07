# WHAT: `and` / `or` over an Optional operand yield the operand that decided,
# as CPython does -- a falsy present value (an empty str, frozenset or list,
# 0, 0j) and not None in its place -- and the operands after it read the name
# as the proof before them left it (`s and s.upper()`, `s is None or ...`),
# printed directly as well as through a local.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defect answered None
# for CPython's `frozenset()` at run time, and the str shapes compiled to a
# lowering failure only for some spellings of the same value; only the printed
# values show both are now the operand itself.

def f(s: str | None, t: str | None) -> None:
    print(s is None or s.upper() == "AB")
    print(s is not None and t is not None and s + t)
    print(s or "default", (s and len(s)) or -1)
    print(repr(s and t and s + t))
    print(s and s.startswith("a"))
def g(n: int | None, xs: list[int] | None) -> None:
    print(n and n + 1, xs and len(xs), xs or [0])
f("ab", "c")
f("", None)
f(None, "x")
g(3, [1]); g(0, []); g(None, None)
total = 0
for v in [1, None, 0, 4]:
    w: int | None = v
    total += (w and w * 2) or 0
print(total)
def a(s: frozenset[int] | None) -> None:
    r = s and 5
    print(repr(r))
def b(z: complex | None) -> None:
    r = z and 5
    print(repr(r))
a(frozenset())
a(None)
a(frozenset([1]))
b(0j)
b(None)
b(1j)
