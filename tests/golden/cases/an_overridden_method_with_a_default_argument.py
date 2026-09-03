# What: a method with a DEFAULT argument, overridden, called through a
# base-typed reference. The dispatcher restates the method's signature to call
# through it, and a default was outside what it would restate -- so the whole
# shape was refused ("'f' is overridden by a subclass of 'Base'") for a method
# Python writes everywhere.
#
# WHY THIS IS RUN: the answer is the body that ran AND the default it filled
# in, and the subclass here gives its parameters DIFFERENT defaults from the
# base's. A dispatcher that restated the base's default would compile and print
# the base's numbers for a subclass instance; the decode is that every arity is
# printed for both classes, so the two are never the same line.
class Base:
    def f(self, a: int = 1, b: int = 2) -> int:
        return a + b


class Sub(Base):
    def f(self, a: int = 5, b: int = 7) -> int:
        return a * b


x: Base = Sub()
print(x.f(), x.f(3), x.f(3, 4))

y: Base = Base()
print(y.f(), y.f(3), y.f(3, 4))

everyone: "list[Base]" = [Base(), Sub()]
print([one.f() for one in everyone])
print([one.f(10) for one in everyone])
