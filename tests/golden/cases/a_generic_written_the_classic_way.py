# The PEP 695 spelling of every program here compiles (cases/generic_classes,
# cases/generic_functions); the spelling every generic before Python 3.12 is
# written in did not, and it failed by FABRICATING a contract. `T = TypeVar("T")`
# made `T` an ordinary name, so an annotation naming it became
# `!py.contract<"builtins.T">` and the program stopped with "print() cannot
# render argument of type builtins.T" -- a sentence about a class nobody wrote.
# The `TypeVar(...)` call itself was "static type builtins.object is not
# callable".
#
# Why execution: the answers are the SPECIALIZED ones. A desugaring that moved
# the parameters but bound them wrongly would compile and hand back the first
# instantiation's answer for the second, which is what the two-instantiation
# lines below are for.
#
# ⛔ Only what the new syntax already supports. `class Pair(Generic[K, V])`
# whose method returns `Pair[V, K]`, a method-level parameter on a
# non-generic class, and `list[list[T]]` all fail here -- and all three fail
# identically when written `class Pair[K, V]`, so the desugaring is level with
# the syntax it desugars to rather than ahead of it.
from typing import Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class Box(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value: T = value

    def get(self) -> T:
        return self.value


class Tagged(Generic[T]):
    def __init__(self, value: T, tag: str) -> None:
        self.value: T = value
        self.tag: str = tag

    def show(self) -> str:
        return self.tag + "=" + str(self.value)


class Base:
    def kind(self) -> str:
        return "base"


class Wrapped(Base, Generic[T]):
    def __init__(self, value: T) -> None:
        self.value: T = value

    def get(self) -> T:
        return self.value


def first(xs: "list[T]") -> T:
    return xs[0]


def pair(a: T, b: U) -> "tuple[T, U]":
    return (a, b)


def repeat(value: T, times: int) -> "list[T]":
    out: "list[T]" = []
    for _ in range(times):
        out.append(value)
    return out


def main() -> None:
    numbers: "Box[int]" = Box(7)
    words: "Box[str]" = Box("q")
    print(numbers.get(), words.get())

    print(Tagged(3, "n").show(), Tagged("x", "s").show())
    print(Wrapped(5).get(), Wrapped(5).kind())

    print(first([1, 2]), first(["a", "b"]))
    print(pair(1, "x"), pair("y", 2))
    print(repeat(0, 3), repeat("z", 2))


main()
