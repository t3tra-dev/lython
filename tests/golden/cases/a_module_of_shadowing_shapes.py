# Helper for a_class_attribute_a_subclass_redeclares_across_modules.
class Shape:
    kind = "shape"
    scale = 1

    def area(self) -> int:
        return 0

    def describe(self) -> str:
        return self.kind + ":" + str(self.area() * self.scale)


class Square(Shape):
    kind = "square"
    scale = 2
    side = 5

    def area(self) -> int:
        return self.side * self.side


class Tilted(Square):
    kind = "tilted"
