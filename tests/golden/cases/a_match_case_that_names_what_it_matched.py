# What: `<pattern> as <name>` binds the whole subject AND matches the pattern,
# so the case body reads both -- running it is the only way to see the two
# bindings are the right ones. The None case is here for the other half: what
# falls past it cannot be None, and the capture after it has to be typed that
# way to be usable at all.
class Point:
    __match_args__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


def describe(point: Point) -> str:
    match point:
        case Point(0, y) as whole:
            return "axis y=" + str(y) + " x=" + str(whole.x)
        case Point(x, y) as whole:
            return "sum=" + str(x + y) + " x=" + str(whole.x)
    return "unreached"


def bump(value: "int | None") -> str:
    match value:
        case None:
            return "none"
        case n:
            return str(n + 1)


def named(value: "int | None") -> str:
    match value:
        case None as nothing:
            return "nothing " + str(nothing)
        case n as num:
            return "num " + str(num * 2)


def first_of(values: "list[int]") -> str:
    match values:
        case [head] as whole:
            return str(head) + " of " + str(len(whole))
        case [head, *rest] as whole:
            return str(head) + "+" + str(len(rest)) + " of " + str(len(whole))
    return "empty"


print(describe(Point(0, 5)), describe(Point(2, 3)))
print(bump(None), bump(4))
print(named(None), named(4))
print(first_of([7]), first_of([1, 2, 3]))
