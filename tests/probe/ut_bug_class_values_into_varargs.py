# WHAT: Class values passed to `*classes: type[Base]`: the lowering stops
#   with "*args source ... must be a lowered object bundle". A
#   `list[type[Base]]` parameter given `[A, B]` works. `unittest.main(TestA,
#   TestB)` needs this.
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- the variadic tuple takes a class value as a list display does.
# CLASSIFICATION @ lang-gaps 699a549f + the fix: 1 正しい
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
