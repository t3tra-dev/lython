# WHAT: Class values passed to `*classes: type[Base]`: the lowering stops
#   with "*args source ... must be a lowered object bundle". A
#   `list[type[Base]]` parameter given `[A, B]` works. `unittest.main(TestA,
#   TestB)` needs this.
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   *args source for function target _ZN8__main__5namesE must be a lowered object bundle
# CPython 3.14 expects: ['A', 'B']
class Base:
    pass


class A(Base):
    pass


class B(Base):
    pass


def names(*classes: type[Base]) -> list[str]:
    return [c.__name__ for c in classes]


print(names(A, B))
