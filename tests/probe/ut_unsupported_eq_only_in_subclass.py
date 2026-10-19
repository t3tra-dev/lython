# WHAT: `a == b` on `Base` receivers where only a SUBCLASS defines __eq__:
#   object.__eq__ returning NotImplemented has no value here.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   '__eq__' is overridden by a subclass of 'Base', so this call cannot be resolved from the static type of the receiver
# CPython 3.14 expects: True False
class Base:
    pass


class Sub(Base):
    def __init__(self, v: int) -> None:
        self.v = v

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Sub) and other.v == self.v


def same(a: Base, b: Base) -> bool:
    return a == b


print(same(Sub(1), Sub(1)), same(Sub(1), Sub(2)))
