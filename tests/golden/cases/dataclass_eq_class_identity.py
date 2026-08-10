# Why execution: the answer was a boolean, not a diagnostic. Two dataclasses
# with the same field compared equal, and a base compared equal to its
# subclass, both with exit 0.
#
# CPython's generated __eq__ opens with `if other.__class__ is self.__class__`
# and returns NotImplemented otherwise, so `==` across classes is False and
# `!=` is True. The synthesized body here compares fields only, and its `other`
# parameter is typed as the class itself, so a cross-class comparison ran the
# field test on operands of two different classes.
from dataclasses import dataclass


@dataclass
class Base:
    x: int


@dataclass
class Sub(Base):
    pass


@dataclass
class Unrelated:
    x: int


@dataclass
class Pair:
    x: int
    y: int


def main() -> None:
    print(Base(1) == Sub(1))
    print(Base(1) != Sub(1))
    print(Base(1) == Unrelated(1))
    print(Base(1) != Unrelated(1))

    print(Base(1) == Base(1))
    print(Base(1) == Base(2))
    print(Base(1) != Base(2))
    print(Pair(1, 2) == Pair(1, 2))
    print(Pair(1, 2) == Pair(1, 3))


main()
