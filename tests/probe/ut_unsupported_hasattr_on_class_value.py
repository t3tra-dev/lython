# WHAT: `hasattr(cls, 'runTest')` on a `type[TestCase]` value: "cannot be
#   answered: 'WithRunTest' derives from it and may define that attribute".
#   The instance form `hasattr(obj, name)` is answered at run time.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   hasattr(x, 'runTest') on 'TestCase' cannot be answered: 'WithRunTest' derives from it and may define that attribute
# CPython 3.14 expects: True False
class TestCase:
    pass


class WithRunTest(TestCase):
    def runTest(self) -> None:
        pass


def has_run_test(cls: type[TestCase]) -> bool:
    return hasattr(cls, 'runTest')


print(has_run_test(WithRunTest), has_run_test(TestCase))
