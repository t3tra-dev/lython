# `...` is what Python writes for a body it does not mean to run: an abstract
# method, a Protocol declaration, a stub a subclass fills in. It was
# "unsupported constant literal" -- the constant emitter has no arm for
# Ellipsis, and this is the one place a program writes one -- so every
# declaration spelled that way was refused.


class Shape:
    def area(self) -> int: ...

    def label(self) -> str:
        return "shape:" + str(self.area())


class Square(Shape):
    def __init__(self, n: int) -> None:
        self.n = n

    def area(self) -> int:
        return self.n * self.n


class Unit(Shape):
    def area(self) -> int:
        return 1


shapes: list[Shape] = [Square(3), Unit()]
print([s.area() for s in shapes])
print([s.label() for s in shapes])


# A stub whose result CAN hold None returns None, exactly as CPython does.
class Hook:
    def before(self) -> None: ...

    def run(self) -> str:
        self.before()
        return "ran"


print(Hook().run(), Hook().before())

# ⛔ DEVIATION, not pinned here because it disagrees with CPython on purpose: a
# stub whose declared result cannot hold None RAISES when it is actually
# called, where CPython returns the None the annotation says the caller cannot
# receive. Measured in tests/probe/wb_calling_an_ellipsis_stub.py.
