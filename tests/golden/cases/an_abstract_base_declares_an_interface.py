# Why execution: the answers are the DISPATCHED ones -- which override runs for
# a base-typed receiver -- and the defect was that the two lines every Python
# interface starts with did not compile at all:
#
#     from abc import ABC, abstractmethod
#     # unsupported import 'abc.abstractmethod'
#
# `@abstractmethod` was already recognized as a decorator; only the import that
# names it was refused. And `ABC` is `object` here: an abstract base declares an
# interface and carries no state or behaviour of its own, which is the whole of
# what it buys a RUNNING program -- dispatching an overridden method from a
# base-typed receiver is something this compiler does for any base.
#
# ⛔ CPython refuses `Base()` while an abstract method is unimplemented; here
# `Base` is an ordinary class and constructing it runs the stub. That is the
# deviation `@overload` and `@final` already carry: the marker constrains a
# CHECKER, and this compiler's checker is its type system.
#
# ⛔ `import abc` -- the module spelling -- is still refused, loudly. The names
# are bound as imports, not as members of a module namespace.
from abc import ABC, abstractmethod


class Shape(ABC):
    @abstractmethod
    def area(self) -> int:
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    def describe(self) -> str:
        return self.name() + ":" + str(self.area())


class Square(Shape):
    def __init__(self, n: int) -> None:
        self.n: int = n

    def area(self) -> int:
        return self.n * self.n

    def name(self) -> str:
        return "square"


class Rect(Shape):
    def __init__(self, w: int, h: int) -> None:
        self.w: int = w
        self.h: int = h

    def area(self) -> int:
        return self.w * self.h

    def name(self) -> str:
        return "rect"


def total(items: "list[Shape]") -> int:
    out = 0
    for item in items:
        out += item.area()
    return out


def widest(items: "list[Shape]") -> str:
    best = items[0]
    for item in items:
        if item.area() > best.area():
            best = item
    return best.name()


def main() -> None:
    shapes: "list[Shape]" = [Square(3), Rect(2, 5)]
    print([s.describe() for s in shapes])
    print(total(shapes), widest(shapes))
    print(isinstance(shapes[0], Shape), isinstance(shapes[0], Square))
    print(isinstance(shapes[1], Square))
    # A base-typed parameter, which is the position the interface exists for.
    print(one(Square(4)), one(Rect(1, 6)))


def one(shape: Shape) -> str:
    return shape.describe()


main()
