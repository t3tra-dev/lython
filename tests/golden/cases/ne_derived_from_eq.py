# What: `!=` on a class that supplies only __eq__ negates THAT __eq__, for
#   every source of __eq__ (user body, dataclass synthesis, enum) and at any
#   field count. The field count carries the weight: the uniform boxed
#   class-id dispatch admits at most five memref operands, and a two-field
#   dataclass's __eq__ takes six (self box + 2 fields, other box + 2 fields),
#   so deriving != at the dispatch instead answered identity and reported
#   `Point(1, 2) != Point(1, 2)` as True.
from dataclasses import dataclass
from enum import Enum


class Tagged:
    def __init__(self, tag: int) -> None:
        self.tag = tag

    def __eq__(self, other: "Tagged") -> bool:
        return self.tag == other.tag


@dataclass
class One:
    a: int


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Wide:
    a: int
    b: int
    c: int
    d: int


class Color(Enum):
    RED = 1
    GREEN = 2


print(Tagged(1) == Tagged(1), Tagged(1) != Tagged(1))
print(Tagged(1) == Tagged(2), Tagged(1) != Tagged(2))
print(One(1) == One(1), One(1) != One(1))
print(Point(1, 2) == Point(1, 2), Point(1, 2) != Point(1, 2))
print(Point(1, 2) == Point(1, 3), Point(1, 2) != Point(1, 3))
print(Wide(1, 2, 3, 4) == Wide(1, 2, 3, 4), Wide(1, 2, 3, 4) != Wide(1, 2, 3, 4))
print(Wide(1, 2, 3, 4) != Wide(1, 2, 3, 5))
print(Color.RED == Color.RED, Color.RED != Color.RED)
print(Color.RED == Color.GREEN, Color.RED != Color.GREEN)
