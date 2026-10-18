# WHAT: a call through a class value -- `c(2)`, `c.make()`, `c.sides()`,
# `c.tag`, `type(self)(...)`, `self.kind()`, `raise exc(msg)` -- runs the class
# the value names at run time, for a class held in a list, a field, a closure,
# a parameter and the result of `type()`.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which constructor, class
# method or class attribute runs is decided by the class the value holds, and
# only the output says which one ran. These calls used to be re-spelled as the
# value's STATIC class, which built an A for a value naming B -- silently, the
# same shape of program printing A's answer twice.
class A:
    tag = "a"

    def __init__(self, n: int) -> None:
        self.n = n

    def describe(self) -> str:
        return "A" + str(self.n)

    @classmethod
    def make(cls) -> "A":
        return cls(0)

    @staticmethod
    def sides() -> int:
        return 0

    def clone(self) -> "A":
        return type(self)(self.n + 1)


class B(A):
    tag = "b"

    def __init__(self, n: int) -> None:
        self.n = n * 10

    def describe(self) -> str:
        return "B" + str(self.n)

    @staticmethod
    def sides() -> int:
        return 3


for c in [A, B]:
    print(c(2).describe(), c.make().describe(), c.sides(), c.tag)
    print(c(1).clone().describe())


class Factory:
    def __init__(self, kind: type[A]) -> None:
        self.kind = kind

    def build(self, n: int) -> A:
        return self.kind(n)


print([Factory(k).build(4).describe() for k in [A, B]])


def maker(cls: type[A]):
    def make(n: int) -> A:
        return cls(n)

    return make


print(maker(B)(5).describe(), maker(A)(5).describe())


def again(x: A) -> str:
    return type(x)(7).describe()


print(again(A(0)), again(B(0)))


class Failure(ValueError):
    pass


def boom(exc: type[Exception], msg: str) -> None:
    raise exc(msg)


for e in [ValueError, KeyError, Failure, LookupError]:
    try:
        boom(e, "x")
    except Exception as caught:
        print(type(caught).__name__, caught, repr(caught))
