# WHAT: `not names and hasattr(cls, 'runTest')` -- two bools -- is refused
#   as an `and` with "no statically representable result type"; each operand
#   alone is accepted. (hasattr on a class value is
#   ut_unsupported_hasattr_on_class_value.)
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- hasattr and callable are typed bool.
# CLASSIFICATION @ lang-gaps 699a549f + the fix: 1 正しい
# CPython 3.14 expects: ['runTest'] ['x']
class TestCase:
    pass


class WithRunTest(TestCase):
    def runTest(self) -> None:
        pass


def names_for(cls: type[TestCase], names: list[str]) -> list[str]:
    if not names and hasattr(cls, 'runTest'):
        return ['runTest']
    return names


print(names_for(WithRunTest, []), names_for(TestCase, ["x"]))
