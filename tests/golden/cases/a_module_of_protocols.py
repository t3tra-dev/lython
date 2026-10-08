# WHAT: a module that declares a protocol and functions over it, imported by
# a_protocol_from_another_module (it prints nothing itself).
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: it is the library half of
# a two-module program; run alone it checks that it compiles and prints
# nothing.
from typing import Protocol


class Shape(Protocol):
    def area(self) -> float: ...
    def name(self) -> str: ...


def biggest(xs: list[Shape]) -> str:
    best = xs[0]
    for x in xs:
        if x.area() > best.area():
            best = x
    return best.name()


def describe(s: Shape) -> str:
    return s.name() + ":" + str(s.area())
