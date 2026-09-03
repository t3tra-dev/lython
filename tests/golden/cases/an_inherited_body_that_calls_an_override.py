# A class inherits its base's method by pointing at the BASE's symbol, and that
# symbol's body typed `self` as the base -- so a call inside it bound statically
# to the base's version. Printing a CONTAINER reaches the body through that
# symbol, and printed the base's answer:
#
#     print([Sq(3)])      # [S(0)]; CPython prints [S(9)]
#
# `print(Sq(3))` beside it was right, because that path INLINES the body at the
# call site where `self` is the receiver's own class. A silent wrong answer, and
# one nothing in the program looks wrong about.


class Shape:
    def area(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "S(" + str(self.area()) + ")"

    def label(self) -> str:
        return "L(" + str(self.area()) + ")"


class Sq(Shape):
    def __init__(self, n: int) -> None:
        self.n = n

    def area(self) -> int:
        return self.n * self.n


print(Sq(3), repr(Sq(3)), Sq(3).label())
print([Sq(3)])
print([Shape(), Sq(2)])
print({"k": Sq(4)})
print((Sq(1), Shape()))
based: list[Shape] = [Shape(), Sq(5)]
print(based)
print(str(based))
