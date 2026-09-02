# What: the template-method shape -- a base method that calls another method on
# `self`, which a subclass overrides. Through a BASE-typed reference the
# override has to win, and through a construction it has to win too; the decode
# is that the two spellings of the same object must print the same word, and
# that word must be the subclass's.
class Shape:
    def __init__(self, size: int) -> None:
        self.size = size

    def name(self) -> str:
        return "shape"

    def describe(self) -> str:
        return self.name() + str(self.size)


class Square(Shape):
    def name(self) -> str:
        return "square"


class Marked(Square):
    def name(self) -> str:
        return "marked"


class Loud(Square):
    def name(self) -> str:
        return super().name().upper()


def describe(s: "Shape") -> str:
    return s.describe()


print(Shape(1).describe(), Square(2).describe(), Marked(3).describe())
print(describe(Shape(1)), describe(Square(2)), describe(Marked(3)))

everything: "list[Shape]" = [Shape(1), Square(2), Marked(3), Loud(4)]
print([s.describe() for s in everything])
print([s.name() for s in everything])

# The base's own method reached through super() resolves the class it names
# rather than dispatching again.
print(Loud(4).describe(), describe(Loud(5)))
