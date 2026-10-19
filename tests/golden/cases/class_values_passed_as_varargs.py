# WHAT: class values passed to a `*classes: type[Base]` parameter -- the
# subclasses, the base, none -- and to a list parameter for comparison; the
# function reads each class's name and builds an instance of each.
#
# WHY THIS IS RUN: which classes arrive is what the call passed. The
# variadic tuple took only object values and refused class values in the
# lowering ("*args source ... must be a lowered object bundle").
class Base:
    def kind(self) -> str:
        return "base"


class A(Base):
    def kind(self) -> str:
        return "a"


class B(Base):
    pass


def names(*classes: type[Base]) -> list[str]:
    return [c.__name__ + ":" + c().kind() for c in classes]


def from_list(classes: list[type[Base]]) -> list[str]:
    return [c.__name__ for c in classes]


print(names(A, B), names(Base), names(), from_list([A, B]))
