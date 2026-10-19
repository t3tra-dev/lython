# WHAT: A @contextmanager method makes an object whose class declares a
#   field in its body only (`params: dict[str, object]`, set later or never)
#   and hands it to an inner @contextmanager around its `yield`. The
#   ownership verifier stops it, "released owned resource from make_sub is
#   used by function return" -- even when nothing calls the method. Without
#   the class-body declaration it compiles. unittest's subTest hands a
#   _SubTest to testPartExecutor this way.
# KIND: バグ (bug) -- loud, from the ownership verifier
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- the post-insertion canonicalizer no longer merges a
#   block whose entry edge lends its arguments a reference of their own
#   (cases/a_call_result_held_across_a_yield_in_a_contextmanager_method).
# CLASSIFICATION @ lang-gaps 326fa275 + the fix: 1 正しい
# CPython 3.14 expects: inside / True
import contextlib
from collections.abc import Iterator


class Outcome:
    def __init__(self) -> None:
        self.success = True

    @contextlib.contextmanager
    def part(self, test_case: TestCase) -> Iterator[None]:
        try:
            yield
        except ValueError:
            self.success = False


class TestCase:
    def __init__(self) -> None:
        self.outcome = Outcome()

    @contextlib.contextmanager
    def subTest(self) -> Iterator[None]:
        sub = make_sub()
        with self.outcome.part(sub):
            yield


class Sub(TestCase):
    params: dict[str, object]


def make_sub() -> Sub:
    return Sub()


case = TestCase()
with case.subTest():
    print("inside")
print(case.outcome.success)
