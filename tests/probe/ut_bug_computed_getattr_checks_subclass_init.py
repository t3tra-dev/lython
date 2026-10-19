# WHAT: `getattr(self, name)()` with a computed name. A subclass's __init__
#   takes a required argument, and its arm is compiled as an inlined call
#   with none ("missing required argument 'description'") instead of raising
#   CPython's TypeError if that name ever arrives.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   missing required argument 'description' for inlined class method
# CPython 3.14 expects: test_a ran / d
class TestCase:
    def __init__(self, methodName: str = "runTest") -> None:
        self._testMethodName = methodName

    def run(self) -> None:
        getattr(self, self._testMethodName)()


class Holder(TestCase):
    def __init__(self, description: str) -> None:
        super().__init__()
        self.description = description


class MyTests(TestCase):
    def test_a(self) -> None:
        print("test_a ran")


MyTests("test_a").run()
print(Holder("d").description)
