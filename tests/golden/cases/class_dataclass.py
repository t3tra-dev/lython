from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int
    label: str = "origin"


@dataclass
class Point3(Point):
    z: int = 0


@dataclass(repr=False)
class Silent:
    v: int

    def __repr__(self) -> str:
        return "<silent>"


p = Point(1, 2)
q = Point(1, 2)
r = Point(3, 4, "far")
print(repr(p))
print(p == q)
print(p == r)
print(p.x + r.y)
print(repr(Point3(5, 6, "high", 7)))
print(repr(Point3(8, 9)))
print(Point3(1, 2) == Point3(1, 2, "origin", 0))
print(Point3(1, 2) == Point3(1, 2, "other", 0))
print(repr(Silent(3)))
print(p)
