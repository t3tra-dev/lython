# What: a nested callable that captures a CLASS. A `type[X]` carries no runtime
# value -- which class it is, is in the type -- so its capture occupies a
# closure slot that stays empty and the callee rebuilds it from its own declared
# closure type. Runtime values, because the question is which class each nested
# call constructs; the same class passed as an argument, held in a field or
# taken as a default has always worked.
from typing import Callable, Iterator


class Widget:
    def __init__(self, n: int) -> None:
        self.n: int = n

    def label(self) -> str:
        return "W" + str(self.n)


def by_def(n: int) -> str:
    cls = Widget
    tag = "t"

    def build(v: int) -> str:
        return tag + cls(v).label()

    return build(n)


def by_lambda(n: int) -> int:
    cls = Widget
    f = lambda v: cls(v).n
    return f(n)


def rows(n: int) -> Iterator[str]:
    cls = Widget

    def build(v: int) -> str:
        return cls(v).label()

    for i in range(n):
        yield build(i)


def returned() -> Callable[[int], str]:
    cls = Widget

    def build(v: int) -> str:
        return cls(v).label()

    return build


class Holder:
    def run(self, n: int) -> str:
        cls = Widget

        def build(v: int) -> str:
            return cls(v).label()

        return build(n)


print(by_def(2))
print(by_lambda(3))
print(list(rows(3)))
print(returned()(4))
print(Holder().run(5))
