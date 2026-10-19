# WHAT: subclasses override a method with other parameters -- fewer, more,
# keyword-only -- and the call through the base-typed value reaches each.
# Where an override cannot take the call, it raises the TypeError CPython
# raises, word for word; where it can, it runs.
#
# WHY THIS IS RUN: the override a call reaches is the receiver's class at run
# time, and so is the TypeError. The dispatch was refused for the whole
# program when any override could not take the call.
class Base:
    def f(self, x: int) -> int:
        return x + 1


class Fewer(Base):
    def f(self) -> int:
        return 0


class More(Base):
    def f(self, x: int, y: int) -> int:
        return x + y


class KeywordOnly(Base):
    def f(self, *, x: int = 5) -> int:
        return x


class Optional(Base):
    def f(self, x: int, y: int = 10) -> int:
        return x + y


items: list[Base] = [Base(), Fewer(), More(), KeywordOnly(), Optional()]
for item in items:
    try:
        print(type(item).__name__, item.f(1))
    except TypeError as e:
        print("TypeError:", e)
    try:
        print(type(item).__name__, item.f(x=2))
    except TypeError as e:
        print("TypeError:", e)
