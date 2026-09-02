# What: a bound METHOD OBJECT taken off a base-typed receiver. `x.f()` goes
# through the dispatcher that tests the runtime class; `m = x.f` built a
# wrapper around the STATIC method instead, so one question asked in two
# spellings answered the base's body for a subclass instance -- silently, with
# the direct call one line over answering correctly.
#
# WHY THIS IS RUN: which body a call reaches is a runtime fact, and a wrapper
# over the wrong body prints an answer of the right shape. The decode is that
# the direct call and every spelling of the method object are printed together:
# a compiler that fixed only the local-variable spelling still prints the
# base's letter for the argument and the list.
from typing import Callable


class Base:
    def f(self) -> str:
        return "B"

    def g(self, n: int) -> str:
        return "B" + str(n)


class Sub(Base):
    def f(self) -> str:
        return "S"

    def g(self, n: int) -> str:
        return "S" + str(n)


def apply(fn: "Callable[[], str]") -> str:
    return fn()


x: Base = Sub()
print(x.f(), (x.f)())
m = x.f
print(m(), apply(x.f), [x.f][0]())
again = m
print(again())
with_argument = x.g
print(with_argument(3))

plain: Base = Base()
print(plain.f(), (plain.f)())
