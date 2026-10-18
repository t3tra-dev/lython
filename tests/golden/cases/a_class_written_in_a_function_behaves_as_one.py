# WHAT: classes written in functions -- in a module function, a method, a
# nested def, a method of one of them -- inherit from each other and from the
# module's classes, call `super()`, name themselves in their own annotations,
# compare, hash and sort through their dunders, take type parameters, and
# shadow a module class of the same name only where the function sees them.
#
# WHY THIS IS RUN: every answer here is a method dispatched, a comparison run,
# or a class matched at run time.
from typing import Iterator


class Shape:
    def area(self) -> int:
        return 0

    def describe(self) -> str:
        return type(self).__name__ + " " + str(self.area())


def shapes() -> list[Shape]:
    class Rect(Shape):
        def __init__(self, w: int, h: int) -> None:
            self.w = w
            self.h = h

        def area(self) -> int:
            return self.w * self.h

        def scaled(self, k: int) -> "Rect":
            return Rect(self.w * k, self.h * k)

    class Square(Rect):
        def __init__(self, side: int) -> None:
            super().__init__(side, side)

        def describe(self) -> str:
            return "square: " + super().describe()

    made: list[Shape] = [Rect(2, 3), Square(4), Rect(1, 1).scaled(5)]
    return made


for s in shapes():
    print(s.describe())


class Version:
    pass


def versions() -> None:
    class Version:
        def __init__(self, major: int, minor: int) -> None:
            self.major = major
            self.minor = minor

        def __lt__(self, other: "Version") -> bool:
            return (self.major, self.minor) < (other.major, other.minor)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Version) and (self.major, self.minor) == (
                other.major,
                other.minor,
            )

        def __hash__(self) -> int:
            return hash((self.major, self.minor))

        def __repr__(self) -> str:
            return "Version(" + str(self.major) + ", " + str(self.minor) + ")"

    vs = [Version(1, 2), Version(0, 9), Version(1, 0)]
    print(sorted(vs))
    print(len({Version(1, 2), Version(1, 2), Version(2, 0)}))
    print(isinstance(vs[0], Version), Version(1, 2) == Version(1, 2))


versions()
print(type(Version()).__qualname__)


def checks() -> None:
    class A:
        pass

    class B(A):
        pass

    a = A()
    b = B()
    print(type(a) is A, type(b) is A, isinstance(b, A), isinstance(a, B))
    print(issubclass(B, A), issubclass(A, B))


checks()


def boxes() -> None:
    class Box[T]:
        def __init__(self, item: T) -> None:
            self.item = item

        def get(self) -> T:
            return self.item

    class IntBox(Box[int]):
        def doubled(self) -> int:
            return self.get() * 2

    print(Box("s").get() + "!", IntBox(21).doubled())


boxes()


def outer() -> int:
    def inner() -> int:
        class Point:
            def __init__(self, x: int) -> None:
                self.x = x

        print(type(Point(0)).__qualname__)
        return Point(41).x + 1

    class Host:
        def make(self) -> int:
            class Part:
                def v(self) -> int:
                    return 7

            print(type(Part()).__qualname__)
            return Part().v()

    return inner() + Host().make()


print(outer())


def items(n: int) -> Iterator[str]:
    class Item:
        def __init__(self, i: int) -> None:
            self.i = i

        def __repr__(self) -> str:
            return "Item(" + str(self.i) + ")"

    for i in range(n):
        yield repr(Item(i))


print(list(items(3)))


def first() -> str:
    class Same:
        def who(self) -> str:
            return "first"

    return Same().who() + " " + type(Same()).__qualname__


def second() -> str:
    class Same:
        def who(self) -> str:
            return "second"

    return Same().who() + " " + type(Same()).__qualname__


print(first(), second())
