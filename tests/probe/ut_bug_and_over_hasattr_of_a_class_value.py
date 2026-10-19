# WHAT: `not names and hasattr(cls, 'runTest')` -- two bools -- is refused
#   as an `and` with "no statically representable result type"; each operand
#   alone is accepted. (hasattr on a class value is
#   ut_unsupported_hasattr_on_class_value.)
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   `and`/`or` over these operand types has no statically representable result type; wrap the operands in explicit comparisons or a conditional expression
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
