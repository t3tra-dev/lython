# Why execution: all four printed the wrong value and compiled cleanly.
#
#   - a @dataclass subclass that declares nothing INHERITS __post_init__, and
#     CPython's generated __init__ calls it. The lookup asked the class's own
#     dict, so the call was dropped and the field kept its default.
#   - comparing two distinct classes was folded to a constant on the reasoning
#     that each has its own __eq__ -- which says nothing about what that
#     __eq__ answers. Only the SYNTHESIZED dataclass one has the class guard
#     the fold assumes.
from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class Base:
    a: int = 0

    def __post_init__(self) -> None:
        self.a = 9


@dataclass
class Derived(Base):
    pass


@dataclass
class P:
    v: int


@dataclass
class Q:
    v: int


class TupleA(NamedTuple):
    v: int


class TupleB(NamedTuple):
    v: int


class HandWrittenX:
    def __eq__(self, other: object) -> bool:
        return True


class HandWrittenY:
    def __eq__(self, other: object) -> bool:
        return True


def main() -> None:
    print(Base().a, Derived().a)
    # A synthesized dataclass __eq__ really does answer False across classes.
    print(P(1) == Q(1), P(1) != Q(1), P(1) == P(1), P(1) == P(2))
    # A NamedTuple's is tuple's, and compares by contents across classes.
    print(TupleA(1) == TupleB(1), TupleA(1) == TupleA(1), TupleA(1) == TupleA(2))
    # A hand-written one answers whatever it likes, so it must be called.
    print(HandWrittenX() == HandWrittenY())


main()
