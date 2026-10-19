# WHAT: `hasattr(cls, 'runTest')` on a `type[TestCase]` value: "cannot be
#   answered: 'WithRunTest' derives from it and may define that attribute".
#   The instance form `hasattr(obj, name)` is answered at run time.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- answered by the run-time class: does it derive from a class
#   that declares the name.
# CLASSIFICATION @ lang-gaps 699a549f + the fix: 1 正しい
# CPython 3.14 expects: True False
class TestCase:
    pass


class WithRunTest(TestCase):
    def runTest(self) -> None:
        pass


def has_run_test(cls: type[TestCase]) -> bool:
    return hasattr(cls, 'runTest')


print(has_run_test(WithRunTest), has_run_test(TestCase))
