# What: a `@staticmethod` an override sits behind, reached two ways -- called
# on a base-typed receiver, and read off one first and called after. Both
# answer the body the RUNTIME class picks, and only running it shows which body
# ran: the call spelling was refused outright and the value spelling answered
# the base's, so the two disagreed with each other as well as with CPython.
#
# The list is base-typed on purpose: it is the shape that makes the receiver's
# class unknowable at the call, and the two-level tail (`Cube` overrides
# nothing) pins that an inheriting subclass lands in its parent's arm.
class Shape:
    @staticmethod
    def sides() -> int:
        return 0

    @staticmethod
    def scale(n: int) -> int:
        return n


class Square(Shape):
    @staticmethod
    def sides() -> int:
        return 4

    @staticmethod
    def scale(n: int) -> int:
        return n * 4


class Cube(Square):
    pass


shapes: list[Shape] = [Shape(), Square(), Cube()]

print("called", [s.sides() for s in shapes])

read: list[int] = []
for s in shapes:
    m = s.scale
    read.append(m(3))
print("read", read)

print("through the class", Shape.sides(), Square.sides(), Cube.sides())
