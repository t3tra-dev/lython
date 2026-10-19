# WHAT: A class with a generic method whose parameters all have defaults. A
#   computed-name call `getattr(self, name)()` builds an arm for every
#   attribute, and the zero-argument arm for that method cannot solve its
#   type parameter ("this call does not determine the type arguments"),
#   though the name never arrives.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- a type parameter is solved from an omitted default, the
#   value the call passes.
# CLASSIFICATION @ lang-gaps c8463f9d + the fix: 1 正しい
# CPython 3.14 expects: test_a ran ValueError
class Case:
    def __init__(self, name: str) -> None:
        self.name = name

    def pick[E: BaseException](self, expected: type[E] = BaseException) -> str:
        return expected.__name__

    def test_a(self) -> None:
        print("test_a ran", self.pick(ValueError))

    def run(self) -> None:
        getattr(self, self.name)()


Case("test_a").run()
