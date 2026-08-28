# WHAT: `f(*t)` for a tuple whose arity its type states -- as a literal, as a
# name, through a parameter, out of a function, out of a list, out of a field,
# and in front of a class constructor.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the expansion decides WHICH
# argument each member becomes, and the members here are deliberately of
# different types in the mixed cases so that a swapped pair is a different
# answer rather than a type error. The uniform ones are the shape that used to
# be refused, and their values say whether the order survived.
#
# ⛔ `tuple[T]` -- what `tuple[T, ...]` becomes once the Ellipsis is dropped --
# states no arity, so `f(*xs)` on one is still refused. That is the only tuple
# spelling left that cannot say how many members it has.
def add(a: int, b: int) -> int:
    return a + b


def mix(a: int, b: str, c: float) -> str:
    return b * a + ":" + str(c)


def add3(a: int, b: int, c: int) -> int:
    return a + b + c


print(add(*(1, 2)))

ys = (3, 4)
print(add(*ys))

annotated: "tuple[int, int]" = (5, 6)
print(add(*annotated))


def through(p: "tuple[int, int]") -> int:
    return add(*p)


print(through((7, 8)))


def produce() -> "tuple[int, int]":
    return (9, 10)


print(add(*produce()))
print(add3(0, *annotated))

t3: "tuple[int, str, float]" = (2, "z", 1.5)
print(mix(*t3))

pairs: "list[tuple[int, int]]" = [(1, 2), (3, 4)]
for pair in pairs:
    print(add(*pair))

counts = {1: 2, 3: 4}
for item in sorted(counts.items()):
    print(add(*item))


class Point:
    def __init__(self, x: int, y: str) -> None:
        self.x = x
        self.y = y

    def show(self) -> str:
        return self.y + str(self.x)


origin: "tuple[int, str]" = (11, "p")
print(Point(*origin).show())


class Holder:
    pair: "tuple[int, str]"

    def __init__(self) -> None:
        self.pair = (12, "q")


print(mix(2, "w", 0.5), Holder().pair[1])
