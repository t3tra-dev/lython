# WHAT: `def make() -> list[Base]: return [Sub()]` -- the return annotation
#   is not the expected type of the list display, which is inferred
#   list[Sub] and then refused ("annotated to return list[Base] but this
#   return gives list[Sub]").
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-21 -- the return check judges a display by its elements, at
#   the declared element type the display is built with.
# CLASSIFICATION @ lang-gaps 91ead2fc + the fix: 1 正しい
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
