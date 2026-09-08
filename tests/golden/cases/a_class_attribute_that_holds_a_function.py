# Why execution: the answer is which FUNCTION ran -- a table of handlers is
# only useful if the entry you name is the one that is called, and an
# inherited-then-overridden entry is what says the lookup walked the MRO
# rather than the first class it found.
#
# `V: Callable[[int], int] = lambda n: n + 1` in a class body could not be read
# at all. Two spellings, two different sentences:
#
#     C.V(1)   # static type type<C> does not provide manifest method 'V'
#     f = C.V  # unsupported static class attribute expression for 'V'
#
# The first is the compiler saying it asked the METHOD table; the second is the
# constant channel, which re-materializes a value per read and has no arm for a
# function. The attribute needed the SLOT the other class attributes have -- and
# the ⛔ note that excluded a Callable from that slot said adding it "moves
# nothing", measured on the CALL spelling alone, where the emitter never got as
# far as the storage.
#
# ⛔ Reading one through an INSTANCE stays refused. `c.V(1)` binds the receiver
# in CPython -- that is what `@staticmethod` exists to opt out of -- and the
# message now says so and names the two spellings that work.
from typing import Callable


def add(a: int, b: int) -> int:
    return a + b


def mul(a: int, b: int) -> int:
    return a * b


class Ops:
    ADD: Callable[[int, int], int] = add
    MUL: Callable[[int, int], int] = mul
    INC: Callable[[int], int] = lambda n: n + 1
    NAME: Callable[[], str] = lambda: "ops"


class Base:
    V: Callable[[int], int] = lambda n: n + 1


class Inherits(Base):
    pass


class Overrides(Base):
    V: Callable[[int], int] = lambda n: n * 10


class Bound:
    def __init__(self, k: int) -> None:
        self.k: int = k

    # The unbound spelling CPython stores in a class dict: called through the
    # CLASS, the receiver is an ordinary first argument.
    SCALE: "Callable[[Bound, int], int]" = lambda self, n: self.k * n

    def use(self, n: int) -> int:
        return Bound.SCALE(self, n) + type(self).SCALE(self, n)


def main() -> None:
    print(Ops.ADD(2, 3), Ops.MUL(2, 3), Ops.INC(9), Ops.NAME())
    print(Base.V(1), Inherits.V(1), Overrides.V(1))

    # Read without calling: the slot hands back the function value.
    picked = Ops.MUL
    print(picked(4, 5))

    table: "list[Callable[[int, int], int]]" = [Ops.ADD, Ops.MUL]
    print([fn(3, 4) for fn in table])

    b = Bound(2)
    print(Bound.SCALE(b, 5), b.use(5))
    print(type(b).SCALE(b, 1))


main()
