# What: the imported half of `a_subclass_imported_from_another_module`. It is a
# case of its own because the golden runner globs every .py here -- running it
# alone declares two classes and prints nothing, which is what its empty
# expectation says.
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


class Circle(Shape):
    def name(self) -> str:
        return "circle"


class Marked(Square):
    pass
