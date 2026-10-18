# Every one of these used to COMPILE and run the BASE's body. The guard
# started at the `a.v()` call site alone, so `len(a)`, `a == b`, `a + 1`,
# `a[0]`, `if a:`, `repr(a)`, `with a:` and an overridden property all walked
# past it -- eleven dunders measured silently wrong while `a.__len__()` on the
# next line was refused.
#
# What each of them does NOW is dispatch: the synthesized dispatcher tests the
# runtime class and runs the body an instance of that class would have run, and
# `a.kind` -- the class ATTRIBUTE spelling of the same question -- joined them.
# So this file's remaining assertion is the one shape no dispatcher can be
# built for: a name only the SUBCLASS declares, where the base's binding is
# object's and object's has no value to fall back to. A `__repr__`, `__str__`
# or `__hash__` there falls back to object's own slot and dispatches; an
# `__eq__` does not, because object's answers NotImplemented.
class A:
    kind: int = 1

    def v(self) -> int:
        return 1

    def __len__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        return True

    def __add__(self, other: int) -> int:
        return 1

    def __getitem__(self, index: int) -> int:
        return 1

    def __repr__(self) -> str:
        return "A"

    @property
    def size(self) -> int:
        return 1


class B(A):
    kind: int = 2

    def v(self) -> int:
        return 2

    def __len__(self) -> int:
        return 2

    def __eq__(self, other: object) -> bool:
        return False

    def __add__(self, other: int) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        return 2

    def __repr__(self) -> str:
        return "B"

    @property
    def size(self) -> int:
        return 2


a: A = B()
print(a.v())
print(len(a))
print(a == a)
print(a + 1)
print(a[0])
print(repr(a))
print(a.size)
print(a.kind)


# The subclass may be the ONLY class that declares it: nothing resolves on the
# base but object's, whose `__eq__` answers NotImplemented for two objects.
class Bare:
    pass


class BareSub(Bare):
    def __eq__(self, other: object) -> bool:
        return True


b: Bare = BareSub()
print(b == b)
