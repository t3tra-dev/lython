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


# ⭐ AND THE SPELLING THAT NAMES THE BASE. `Base.__init__(self, ...)` is the
# same call `super().__init__(...)` makes, and for a SOURCE base it has always
# worked -- the unbound method binds no receiver and takes it as the first
# argument. For a builtin base there is no method to bind, so an exception
# subclass forwarding its message this way was told that
# `type<builtins.Exception>` "has manifest method '__init__' but no signature
# that accepts (AppError, str)" -- the compiler saying it asked the class
# object. Real code writes it both ways.
class AppError(Exception):
    def __init__(self, code: int, message: str) -> None:
        Exception.__init__(self, message)
        self.code: int = code


class BadInput(ValueError):
    def __init__(self, field: str) -> None:
        ValueError.__init__(self, "bad " + field)
        self.field: str = field


class Wrapped(AppError):
    def __init__(self, message: str) -> None:
        AppError.__init__(self, 500, "wrapped: " + message)


class Silent(Exception):
    def __init__(self) -> None:
        Exception.__init__(self)
        self.seen: bool = True


def named_base() -> None:
    try:
        raise AppError(400, "negative")
    except AppError as e:
        print("app", e.code, e)
    try:
        raise BadInput("age")
    except ValueError as e:
        print("value", str(e))
    try:
        raise Wrapped("inner")
    except AppError as e:
        print("wrapped", e.code, e)
    try:
        raise Silent()
    except Silent as e:
        print("silent", e.seen, str(e) == "")


named_base()
