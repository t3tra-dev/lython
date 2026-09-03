# A class attribute the subclass shadows, read through a base-typed receiver.
# The value has to come from the receiver's RUNTIME class -- reading the base's
# binding is what a program would never notice going wrong, since the base's
# value is a plausible one.


class Shape:
    kind = "shape"
    sides = 0
    closed = True
    ratio = 1.0

    def describe(self) -> str:
        return self.kind + ":" + str(self.sides)


class Square(Shape):
    kind = "square"
    sides = 4
    closed = False
    ratio = 2.5


class Tilted(Square):
    kind = "tilted"
    ratio = 3.5


def render(s: Shape) -> str:
    return s.describe() + "|" + str(s.closed) + "|" + str(s.ratio)


for shape in [Shape(), Square(), Tilted()]:
    print(render(shape))

# Through the class itself the answer has never been in doubt, and must not
# change: `Shape.kind` is the base's own binding.
print(Shape.kind, Square.kind, Tilted.kind)
