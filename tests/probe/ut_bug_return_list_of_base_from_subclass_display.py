# WHAT: `def make() -> list[Base]: return [Sub()]` -- the return annotation
#   is not the expected type of the list display, which is inferred
#   list[Sub] and then refused ("annotated to return list[Base] but this
#   return gives list[Sub]").
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   function is annotated to return !py.contract<"builtins.list", [!py.contract<"Base">]> but this return gives !py.contract<"builtins.list", [!py.contract<"Sub">]>
# CPython 3.14 expects: ['sub']
class Base:
    def name(self) -> str:
        return "base"


class Sub(Base):
    def name(self) -> str:
        return "sub"


def make() -> list[Base]:
    return [Sub()]


print([b.name() for b in make()])
