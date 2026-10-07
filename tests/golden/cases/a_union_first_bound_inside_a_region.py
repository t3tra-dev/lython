# WHAT: a name whose first binding is inside a loop or a match, and whose
# value is a union -- `T | None` from a ternary or `dict.get`, `int | str`,
# `int | float`, a class or None -- is read after the region with the value
# its last trip or arm bound, and a loop that never ran leaves it unbound.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the name lives in a slot
# whose first, unbound contents were a zeroed union; the defect was a refcount
# abort at run time on the slot's first write, so the values printed after
# each region, over repeated trips, are the claim.

from typing import Optional


def use(flag: int) -> int:
    w: Optional[int] = 3
    match flag:
        case 1:
            v = w
        case _:
            v = w
    return 0 if v is None else v


print(use(1), use(2))

class P:
    def __init__(self, n: int) -> None:
        self.n = n
    def __repr__(self) -> str:
        return f"P({self.n})"
def pick(i: int) -> None:
    for k in range(i):
        a = k if k % 2 == 0 else str(k)
        b = [k] if k > 1 else None
        c = P(k) if k > 0 else None
        d = (k, "t") if k != 0 else None
        e = True if k == 1 else None
        f = b"x" if k != 0 else "y"
        g = {"k": k} if k == 2 else None
        h = 1.5 if k == 3 else k
    print(a, b, c, d, e, f, g, h)
pick(1)
pick(3)
pick(4)
def unbound() -> None:
    for k in range(0):
        z = k if k != 0 else None
    try:
        print(z)
    except UnboundLocalError as err:
        print("unbound", err)
unbound()
total = 0
for k in range(100):
    m = [k] if k % 3 != 0 else None
    if m is not None:
        total += m[0]
print(total, m)

for w in ["a", ""]:
    got = 1 if w else 0.5
print(got)

dd: dict[str, int] = {"a": 1}
for w in ["a", "zz"]:
    got2 = dd.get(w)
print(got2)

def narrowed(xs: list[int | None]) -> None:
    for x in xs:
        if x is not None:
            v = x
    print(v)
    if v is not None:
        print(v + 1)
narrowed([1, None, 3])
