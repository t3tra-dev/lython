from typing import NamedTuple


class Point(NamedTuple):
    x: int
    y: int


class Labelled(NamedTuple):
    name: str
    weight: float
    tags: int = 0


p = Point(1, 2)
q = Point(1, 2)
r = Point(3, 4)

# Field access by name, and the CPython repr spelling.
print(p.x, p.y)
print(p)
print(repr(r))

# Equality is field-wise.
print(p == q, p == r)

# Defaults follow the dataclass field order rules.
a = Labelled("a", 1.5)
b = Labelled("b", 2.5, 7)
print(a)
print(b)
print(a.name, a.weight, a.tags)
print(b.tags)


def midpoint(lhs: Point, rhs: Point) -> Point:
    return Point((lhs.x + rhs.x) // 2, (lhs.y + rhs.y) // 2)


print(midpoint(p, r))
