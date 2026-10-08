# WHAT: a base class and a function over it, imported by
# a_subclass_in_main_reached_from_a_library (it prints nothing itself).
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: it is the library half of
# a two-module program; run alone it checks that it compiles and prints
# nothing.
class Shape:
    def area(self) -> float:
        return 0.0


def total(xs: list[Shape]) -> float:
    t = 0.0
    for x in xs:
        t += x.area()
    return t
