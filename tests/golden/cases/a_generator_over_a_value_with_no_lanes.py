# What: a generator whose PARAMETER is a value with no runtime lanes -- a
# `type[X]`, which decides which class it is from its own type, or `None`,
# which carries nothing at all. The frame stores each argument so the drop
# finalizer can resume without call-site evidence, and a zero-width one has
# nothing to store. Runtime values, because the question is which class each
# resume constructs; the same parameter on a plain FUNCTION and the same class
# as a LOCAL inside the generator have always worked.
#
# ⛔ The class is not subclassed here on purpose: a `type[X]` parameter reached
# with a SUBCLASS is a separate refusal ("whose class is subclassed in this
# program, so which class it names is not decided by its type"), and it is not
# what this case is about.
from typing import Callable, Iterator


class Widget:
    def __init__(self, n: int) -> None:
        self.n: int = n

    def label(self) -> str:
        return "W" + str(self.n)


def rows(cls: type[Widget], n: int) -> Iterator[str]:
    for i in range(n):
        yield cls(i).label()


def flags(cls: type[Widget], n: int) -> Iterator[bool]:
    for i in range(n):
        yield cls(i).n % 2 == 0


def missing(v: None, n: int) -> Iterator[str]:
    for i in range(n):
        yield "n" if v is None else str(i)


def shadowed(Widget: type[Widget], n: int) -> Iterator[str]:
    for i in range(n):
        yield Widget(i).label()


class Factory:
    def rows(self, cls: type[Widget], n: int) -> Iterator[str]:
        for i in range(n):
            yield cls(i).label()


# The same class held in a FIELD: `self.cls(i)` is a constructor, not a method
# named `cls`, and only the yield walk asks the inference that question.
class Held:
    def __init__(self, cls: type[Widget]) -> None:
        self.cls: type[Widget] = cls

    def rows(self, n: int) -> Iterator[str]:
        for i in range(n):
            yield self.cls(i).label()


# A closure over a `type[X]` PARAMETER, returned out of the function that
# built it: its capture rides out as a lane that carries nothing.
def factory(cls: type[Widget]) -> Callable[[int], str]:
    def build(v: int) -> str:
        return cls(v).label()

    return build


print(list(rows(Widget, 3)))
print(list(flags(Widget, 4)))
print(list(missing(None, 2)))
print(list(shadowed(Widget, 2)))
print(list(Factory().rows(Widget, 2)))
print(list(Held(Widget).rows(2)))
print(factory(Widget)(9))
