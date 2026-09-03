# The narrowing an `isinstance` proves, spent inside a conditional EXPRESSION.
# The `if` statement spelling of each of these compiles; the one-line spelling
# used to be refused, because the expression knew how to unwrap a union member
# and nothing else.


class Shape:
    pass


class Square(Shape):
    sides = 4

    def area(self, n: int) -> int:
        return n * n


class Circle(Shape):
    sides = 0


def area_of(s: Shape) -> int:
    return s.area(3) if isinstance(s, Square) else -1


def sides_of(s: Shape) -> int:
    return -1 if not isinstance(s, Square) else s.sides


shapes: list[Shape] = [Shape(), Square(), Circle()]
print([area_of(s) for s in shapes])
print([sides_of(s) for s in shapes])
print([(s.sides if isinstance(s, Square) else 0) for s in shapes])


# The union spelling the expression already handled must keep working, and the
# name must be its old self on the other side of the expression.
def label(v: int | None) -> str:
    out = str(v * 2) if v is not None else "none"
    return out + "/" + ("some" if v is not None else "empty")


print(label(3), label(None))
