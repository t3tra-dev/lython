# Every one of these used to COMPILE and run the BASE's body. The guard
# started at the `a.v()` call site alone, so `len(a)`, `a == b`, `a + 1`,
# `a[0]`, `if a:`, `repr(a)`, `with a:` and an overridden property all walked
# past it -- eleven dunders measured silently wrong while `a.__len__()` on the
# next line was refused. There is no dynamic dispatch to fall back to, so all
# of them are refused where the hierarchy is visible.
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
