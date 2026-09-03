# The class a `case` pattern proves, spent on the subject's own name and on an
# `as` capture. The pattern IS the test -- there is no `if` to hang the
# narrowing on -- so the body could not use what the case had just established.


def classify(v: int | str | None) -> str:
    match v:
        case None:
            return "none"
        case int():
            return "i" + str(v + 1)
        case str():
            return "s" + v.upper()
    return "?"


print(classify(None), classify(1), classify("a"))


def tagged(v: int | str) -> str:
    match v:
        case int() as n:
            return "n" + str(n * 2)
        case str() as s:
            return "s" + s.title()
    return "?"


print(tagged(4), tagged("ab"))


class Shape:
    pass


class Square(Shape):
    sides = 4

    def area(self, n: int) -> int:
        return n * n


class Circle(Shape):
    sides = 0


def describe(s: Shape) -> str:
    match s:
        case Square() as sq:
            return str(sq.sides) + "/" + str(sq.area(3))
        case Circle():
            return str(s.sides)
        case _:
            return "-"


print([describe(s) for s in [Shape(), Square(), Circle()]])


# The name goes back to what it was after the match, and a body that REBINDS
# it keeps its own value.
def rebinds(v: int | str) -> str:
    match v:
        case int():
            v = 99
            return str(v)
        case str():
            return v
    return "?"


print(rebinds(1), rebinds("z"))
