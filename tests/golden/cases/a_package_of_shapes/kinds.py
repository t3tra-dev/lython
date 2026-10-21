from . import shape
from .shape import Shape


class Square(shape.Shape):
    def name(self) -> str:
        return "square"


class Circle(Shape):
    def name(self) -> str:
        return "circle"
