# WHAT: classes in the main module satisfy a protocol declared in another
# module, and the other module's functions dispatch on them -- whose bodies
# are emitted before the main module's classes exist.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which class's method a
# library function reaches is a run time fact; the dispatcher used to be
# refused ("'Tri.area' is used before 'Tri' is defined") because its body was
# built before the class it names.
from a_module_of_protocols import Shape, biggest, describe
import a_module_of_protocols


class Square:
    def __init__(self, s: float) -> None:
        self.s = s

    def area(self) -> float:
        return self.s * self.s

    def name(self) -> str:
        return "square"


class Tri:
    def __init__(self, b: float, h: float) -> None:
        self.b = b
        self.h = h

    def area(self) -> float:
        return self.b * self.h / 2

    def name(self) -> str:
        return "tri"


xs: list[Shape] = [Square(2.0), Tri(3.0, 4.0)]
print(biggest(xs))
print([describe(x) for x in xs])
print(a_module_of_protocols.describe(Tri(1.0, 1.0)))
