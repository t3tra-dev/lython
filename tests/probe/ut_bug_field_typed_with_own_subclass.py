# WHAT: A base class declares a field typed with its own SUBCLASS
#   (`self.sub: Sub | None`). The subclass then loses the fields the base's
#   __init__ sets ("class Sub has no field 'name'"). TestCase holds its
#   current `_SubTest | None` this way.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- a subclass named in its base's body is no longer emitted
#   ahead of the base.
# CLASSIFICATION @ lang-gaps 699a549f + the fix: 1 正しい
# CPython 3.14 expects: x s
class Base:
    def __init__(self, name: str = "x") -> None:
        self.name = name
        self.sub: Sub | None = None


class Sub(Base):
    pass


b = Base()
b.sub = Sub("s")
print(b.name, b.sub.name if b.sub is not None else "-")
