# A conjunction proves one thing per operand, and the BODY gets to keep all of
# them. Only the first name used to narrow -- inside the condition the later
# operands were already narrowed, which is why `and b.only()` worked there and
# the body did not.


def add(a: int | None, b: int | None) -> int:
    if a is not None and b is not None:
        return a + b
    return -1


print(add(1, 2), add(1, None), add(None, None))


def guarded(a: int | None, b: int | None) -> int:
    if a is None or b is None:
        return -1
    return a * b


print(guarded(3, 4), guarded(None, 4))


def three(a: int | None, b: int | None, c: int | None) -> int:
    return a + b + c if a is not None and b is not None and c is not None else 0


print(three(1, 2, 3), three(1, None, 3))


class Shape:
    pass


class Square(Shape):
    sides = 4


class Circle(Shape):
    radius = 2


def pair(x: Shape, y: Shape) -> int:
    if isinstance(x, Square) and isinstance(y, Circle):
        return x.sides + y.radius
    return -1


print(pair(Shape(), Shape()), pair(Square(), Circle()))


def loop(a: int | None, b: int | None) -> int:
    total = 0
    while a is not None and b is not None:
        total = a + b
        a = None
    return total


print(loop(5, 6), loop(None, 6))
