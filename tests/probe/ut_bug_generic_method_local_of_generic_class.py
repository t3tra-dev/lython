# WHAT: Inside a generic METHOD, `context = Ctx[E](expected)` assigned to a
#   local specializes Ctx at the unsolved E as well ("class value has no
#   type object for builtins.E" from the lowering) -- even when the method
#   is never called, so importing a module that has one is enough. Returning
#   `Ctx[E](expected)` directly, or the same local in a generic FUNCTION,
#   compiles. unittest's assertRaises builds its context manager this way.
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   class value has no type object for builtins.E
# CPython 3.14 expects: False
class Ctx[E: BaseException]:
    def __init__(self, expected: type[E]) -> None:
        self.expected = expected

    def check(self, exc_type: type[BaseException]) -> bool:
        return issubclass(exc_type, self.expected)


class TestCase:
    def raises[E: BaseException](self, expected: type[E]) -> bool:
        context = Ctx[E](expected)
        return context.check(KeyError)


print(TestCase().raises(ValueError))
