# WHAT: A loader builds `cls(name)` for `cls: type[TestCase]`. One subclass
#   has an __init__ that takes other arguments, so the call is refused for
#   all ("missing required argument 'extra'"), though that class is never
#   passed.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   missing required argument 'extra' for inlined class method
# CPython 3.14 expects: ['test_a'] d
class TestCase:
    def __init__(self, methodName: str = "runTest") -> None:
        self.methodName = methodName


class Holder(TestCase):
    def __init__(self, description: str, extra: int) -> None:
        super().__init__()
        self.description = description


class MyTests(TestCase):
    def test_a(self) -> None:
        pass


def load(cls: type[TestCase], names: list[str]) -> list[TestCase]:
    return [cls(n) for n in names]


print([t.methodName for t in load(MyTests, ["test_a"])], Holder("d", 1).description)
