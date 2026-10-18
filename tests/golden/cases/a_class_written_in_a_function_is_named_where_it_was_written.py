# WHAT: a class written in a function is named by where it was written --
# `__qualname__` "make.<locals>.Local", `__name__` "Local", `__module__`
# "__main__" -- whether the class is known where it is asked or only at run
# time, and so is every text that prints it: the default repr, the class's
# repr, an enum member, a NamedTuple and a dataclass.
#
# WHY THIS IS RUN: `type(b).__qualname__` for a value whose static type is its
# base is read off the class's type object at run time, which has to say where
# its module ends; and the reprs are text the running program builds.
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple


class Base:
    pass


def make() -> Base:
    class Local(Base):
        pass

    print(Local.__name__, Local.__qualname__, Local.__module__)
    return Local()


def describe(b: Base) -> str:
    t = type(b)
    return t.__name__ + " " + t.__qualname__ + " " + t.__module__


b = make()
print(describe(b))
print(describe(Base()))
r = repr(b)
print(r[: r.index(" object at")])
print(repr(type(b)))


class Suite:
    def test_names(self) -> None:
        class Helper:
            pass

        class Color(Enum):
            RED = 1

        class Pt(NamedTuple):
            x: int

        @dataclass
        class Rec:
            v: int

        class Problem(Exception):
            pass

        print(type(Helper()).__qualname__)
        print(Color.RED, repr(Color.RED))
        print(Pt(1), Rec(2))
        try:
            raise Problem("bad")
        except Problem as e:
            print(type(e).__name__, type(e).__qualname__, e)


Suite().test_names()
