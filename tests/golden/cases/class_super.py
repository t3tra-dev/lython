class Shape:
    def __init__(self, name: str) -> None:
        self.name = name

    def describe(self) -> str:
        return "shape " + self.name


class Circle(Shape):
    def __init__(self, radius: float) -> None:
        super().__init__("circle")
        self.radius = radius

    def describe(self) -> str:
        return super().describe() + " r=" + str(self.radius)


class Tagged:
    def tag(self) -> str:
        return "base"


class Left(Tagged):
    def tag(self) -> str:
        return "L(" + super().tag() + ")"


class Right(Tagged):
    def tag(self) -> str:
        return "R(" + super().tag() + ")"


class Both(Left, Right):
    def tag(self) -> str:
        return "B(" + super().tag() + ")"


c = Circle(2.0)
print(c.name)
print(c.describe())
print(Both().tag())
print(Left().tag())
print(super(Left, Both()).tag())
