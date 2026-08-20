# Which body a method call runs was decided from the STATIC receiver type, so
# an override behind a base-typed reference was refused outright ("'name' is
# overridden by a subclass of 'A', so this call cannot be resolved from the
# static type of the receiver") -- and that took every base-typed collection,
# parameter and declared binding with it. Must run: the refusal is what
# regresses, but only the printed values say the RIGHT body ran, and a
# dispatcher that always picked the base would compile just as quietly.


class Shape:
    def name(self) -> str:
        return "shape"

    def sides(self) -> int:
        return 0

    def label(self, prefix: str) -> str:
        return prefix + self.name()


class Rect(Shape):
    def name(self) -> str:
        return "rect"

    def sides(self) -> int:
        return 4


class Square(Rect):
    def name(self) -> str:
        return "square"


class Circle(Shape):
    def name(self) -> str:
        return "circle"


# A base-typed collection, the shape that made this unavoidable.
shapes: list[Shape] = [Shape(), Rect(), Square(), Circle()]
for s in shapes:
    print(s.name(), s.sides())

# A base-typed parameter.
def describe(s: Shape) -> str:
    return s.name() + "/" + str(s.sides())


print(describe(Square()), describe(Shape()), describe(Circle()))

# A base-typed binding of a subclass value.
one: Shape = Rect()
print(one.name(), one.sides())

# A method with a parameter, dispatched the same way.
print(describe(Rect()), Rect().label("a "), describe(Square()))
labels: list[str] = []
for s in shapes:
    labels.append(s.label("<"))
print(labels)

# Most-derived first: Square overrides name() but inherits Rect's sides().
sq: Shape = Square()
print(sq.name(), sq.sides())

# A base method calling a virtual one on ITSELF, reached through the base type.
holder: Shape = Circle()
print(holder.label("["))

# An OPERATOR reaches a method too, and eleven dunders were measured silently
# wrong on a base-typed receiver before the refusal existed. `len(b)` and
# `b.__len__()` are one method, so they take one dispatcher.
class Bag:
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __str__(self) -> str:
        return "bag"


class Full(Bag):
    def __len__(self) -> int:
        return self.n * 2

    def __str__(self) -> str:
        return "full"


bags: list[Bag] = [Bag(2), Full(3), Bag(0)]
for b in bags:
    print(len(b), str(b), bool(b))

# The operator spellings reach the same dispatcher: an arithmetic dunder, a
# comparison, print()'s __str__ and repr()'s __repr__ each have their own site.
class Money:
    def __init__(self, v: int) -> None:
        self.v = v

    def __add__(self, other: int) -> int:
        return self.v + other

    def __eq__(self, other: object) -> bool:
        return False

    def __repr__(self) -> str:
        return "Money"

    def __str__(self) -> str:
        return "money"


class Doubled(Money):
    def __add__(self, other: int) -> int:
        return self.v * other

    def __eq__(self, other: object) -> bool:
        return True

    def __repr__(self) -> str:
        return "Doubled"

    def __str__(self) -> str:
        return "doubled"


monies: list[Money] = [Money(3), Doubled(3)]
for m in monies:
    print(m + 2, m == m, repr(m))
for m in monies:
    print(m)

# A dispatched body that needs THE SAME dispatcher: `Wrap.size` calls `.size()`
# on a base-typed value, and that call is emitted while the dispatcher for
# Node.size is still being built. Its memo entry therefore has to be complete
# -- symbol AND callable -- before the body is emitted; an entry carrying only
# the symbol answered "no dispatcher yet" and the program came back refused.
class Node:
    def size(self) -> int:
        return 1


def a_node() -> Node:
    return Node()


class Wrap(Node):
    def size(self) -> int:
        return a_node().size() + 1


nodes: list[Node] = [Node(), Wrap()]
for n in nodes:
    print(n.size())
