# What: a `@classmethod` reached through a base-typed receiver. Two defects in
# one mechanism, and only running it shows which body ran with which `cls`:
#
#   an OVERRIDDEN classmethod was refused -- "'tag' is overridden by a subclass
#     of 'Base', so this call cannot be resolved from the static type";
#   an INHERITED one was worse: `class Sub(Base): pass` answered `Base` where
#     CPython answers `Sub`, silently, because nothing redeclared anything and
#     the override test could not see it.
#
# `cls` binds the RUNTIME class, so the candidate set is every SUBCLASS rather
# than the redeclaring slice of it, and each arm calls through the candidate
# CLASS -- which is what binds `cls` to it. `Leaf` below redeclares nothing and
# still needs its own arm.
#
# ⛔ AND ONLY WHERE THE BODY READS `cls`. Which class is bound is unobservable
# in a body that never mentions it, and asking for a dispatcher there would
# refuse the shapes the arms cannot restate -- a `*args` classmethod runs
# correctly and keeps doing so.
#
# ⛔ The VALUE spelling (`m = x.tag`) is still refused: carrying `cls` into a
# value is a bound object over a class, which nothing here builds.
class Base:
    @classmethod
    def tag(cls) -> str:
        return "base:" + cls.__name__

    @classmethod
    def make(cls, n: int) -> str:
        return cls.__name__ + "/" + str(n)

    @classmethod
    def build(cls) -> "Base":
        return cls()

    def name(self) -> str:
        return "base"


class Sub(Base):
    @classmethod
    def tag(cls) -> str:
        return "sub:" + cls.__name__

    def name(self) -> str:
        return "sub"


class Leaf(Sub):
    pass


class Quiet(Base):
    pass


class Counted(Base):
    # A body that never reads `cls`: no dispatcher is needed, and asking for
    # one would refuse the vararg the arms cannot restate.
    @classmethod
    def total(cls, *values: int) -> int:
        out = 0
        for v in values:
            out += v
        return out


def main() -> None:
    xs: "list[Base]" = [Base(), Sub(), Leaf(), Quiet()]
    print([x.tag() for x in xs])
    print([x.make(2) for x in xs])
    # A classmethod that CONSTRUCTS through `cls`: the instance it returns has
    # to be the runtime class's, which the arm decides.
    print([x.build().name() for x in xs])
    # Through the class, which was always exact.
    print(Base.tag(), Sub.tag(), Leaf.tag(), Quiet.tag())
    counters: "list[Counted]" = [Counted(), Counted()]
    print([c.total(1, 2, 3) for c in counters])


main()
