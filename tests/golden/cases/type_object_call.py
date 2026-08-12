# Why execution: the construction happens at run time and the value it
# produces is the whole point. `cls(v)` inside a classmethod and `t = int;
# t()` both reached the lowering as "calling a type object held in a value is
# not supported ...; construct through the type name directly" -- but the
# class IS in the type (py.type<C>), and cls is bound to the RECEIVER's class,
# so Child.make() has to build a Child.
class Base:
    def __init__(self, v: int) -> None:
        self.v = v

    @classmethod
    def make(cls, v: int) -> "Base":
        return cls(v)

    @classmethod
    def zero(cls) -> "Base":
        return cls(0)


class Child(Base):
    pass


def main() -> None:
    print(Base.make(1).v, Child.make(2).v, Base.zero().v)
    t = int
    print(t(), t())
    u = str
    print(u(5), u(1.5))
    b = bool
    print(b())


main()
