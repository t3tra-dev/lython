# WHAT: a subclass defined in the main module overrides a method that a
# function in an imported module calls through the base type.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which body the library's
# call reaches is a run time fact; it was refused ("'Sq.area' is used before
# 'Sq' is defined") because the dispatcher was built while the library was
# emitted, before the main module's classes.
from a_module_of_a_base_and_its_total import Shape, total


class Sq(Shape):
    def area(self) -> float:
        return 4.0


print(total([Sq(), Shape(), Sq()]))
