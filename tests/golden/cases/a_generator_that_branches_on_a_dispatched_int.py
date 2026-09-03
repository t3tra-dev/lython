# A generator whose branch condition compares a value returned by a dispatched
# call. The resume is compiled as a primitive-i64 clone, and a clone that cannot
# vouch for its raw operands parks "I cannot say" and answers false -- which is
# sound only because the CALLER re-runs the boxed original. A resume's caller is
# the runtime's `next`, which has no boxed original, so the false was branched
# on and observed:
#
#     print(list(g(Sq(3))))     # printed [0]; CPython prints [1]


from typing import Iterator


class Shape:
    def area(self) -> int:
        return 0


class Sq(Shape):
    def __init__(self, n: int) -> None:
        self.n = n

    def area(self) -> int:
        return self.n * self.n


def flagged(s: Shape) -> Iterator[int]:
    if s.area() > 0:
        yield 1
    else:
        yield 0


print(list(flagged(Shape())), list(flagged(Sq(3))))


def positives(items: list[Shape]) -> Iterator[int]:
    for s in items:
        if s.area() > 0:
            yield s.area()


shapes: list[Shape] = [Shape(), Sq(3), Sq(2)]
print(list(positives(shapes)))


def ternary(items: list[Shape]) -> Iterator[int]:
    for s in items:
        yield 1 if s.area() > 0 else 0


print(list(ternary(shapes)))
