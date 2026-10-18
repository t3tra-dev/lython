# WHAT: a decorator on a method -- generic over the class it decorates, a
# factory, stacked, imported -- makes the method what the decorator returned:
# called through an instance, through `self` inside the class, as a method
# value, by a computed name, inherited, overridden with `super()` reaching
# the decorated base, and through the base type; the decorator runs once,
# and `dir`/`hasattr` show the method's name and no name the rewrite made.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answers are the
# wrappers', and the log of what ran and when is made at run time. Every
# decorated method was refused ("decorator 'doubled' is not supported").
from contextlib import contextmanager
from typing import Callable, Iterator

import a_module_of_decorators

log: list[str] = []


class Base:
    def __init__(self, k: int) -> None:
        self.k = k


def doubled[S: Base](f: Callable[[S, int], int]) -> Callable[[S, int], int]:
    log.append("doubled applied")

    def wrapper(self: S, x: int) -> int:
        log.append("doubled " + str(x))
        return f(self, x) * 2
    return wrapper


def counted[S: Base](f: Callable[[S, int], int]) -> Callable[[S, int], int]:
    log.append("counted applied")
    calls = [0]

    def wrapper(self: S, x: int) -> int:
        calls[0] += 1
        log.append("call #" + str(calls[0]))
        return f(self, x)
    return wrapper


class Calc(Base):
    @doubled
    def scale(self, x: int) -> int:
        return x * self.k

    @counted
    @doubled
    def both(self, x: int) -> int:
        return x + self.k

    @a_module_of_decorators.mark
    def marked(self, x: int = 3) -> int:
        return x - self.k

    @counted
    def fact(self, x: int) -> int:
        return 1 if x <= 1 else x * self.fact(x - 1)

    def uses(self) -> int:
        return self.scale(1) + self.both(1)

    @contextmanager
    def held(self) -> Iterator[int]:
        yield self.k


class Sub(Calc):
    def scale(self, x: int) -> int:
        return super().scale(x) + 1000


c = Calc(3)
print(c.scale(2), c.both(2), c.marked(), c.marked(10), c.uses())
print(c.fact(4))
f = c.scale
print(f(5), getattr(c, "scale")(6))
s = Sub(2)
print(s.scale(1), s.both(1))


def through(b: Calc) -> int:
    return b.scale(1)


print(through(c), through(s))
print([n for n in dir(Calc) if not n.startswith("_")])
print(hasattr(c, "scale"), hasattr(c, "held"), hasattr(c, "held$cmgen"))
print(log)
print(a_module_of_decorators.applied)
