# WHAT: a test framework's decorators on test methods -- `@skip(reason)` and
# `@skipIf(condition, reason)` as an instance whose generic `__call__` wraps
# the method, `@expectedFailure` as a generic function -- and a runner that
# finds the tests by `dir` and calls each by its computed name: skipped,
# skipped by a condition, not skipped, an expected failure, an unexpected
# success and a plain failure.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which outcome each test
# has is decided by what its wrapper raises at run time. A decorator on a
# method was refused outright ("decorator 'skip' is not supported").
from typing import Callable


class SkipTest(Exception):
    pass


class _ExpectedFailure(Exception):
    pass


class _UnexpectedSuccess(Exception):
    pass


class TestCase:
    def run(self, name: str) -> str:
        try:
            getattr(self, name)()
            return name + ": ok"
        except SkipTest as e:
            return name + ": skipped (" + str(e) + ")"
        except _ExpectedFailure:
            return name + ": expected failure"
        except _UnexpectedSuccess:
            return name + ": unexpected success"
        except AssertionError as e:
            return name + ": FAIL " + str(e)


class skip:
    def __init__(self, reason: str, active: bool = True) -> None:
        self.reason = reason
        self.active = active

    def __call__[T: TestCase](self, f: Callable[[T], None]) -> Callable[[T], None]:
        if not self.active:
            return f
        reason = self.reason

        def skip_wrapper(test: T) -> None:
            raise SkipTest(reason)
        return skip_wrapper


def skipIf(condition: bool, reason: str) -> skip:
    return skip(reason, condition)


def expectedFailure[T: TestCase](f: Callable[[T], None]) -> Callable[[T], None]:
    def wrapper(test: T) -> None:
        try:
            f(test)
        except AssertionError:
            raise _ExpectedFailure()
        raise _UnexpectedSuccess()
    return wrapper


class MyTest(TestCase):
    @skip("not today")
    def test_skipped(self) -> None:
        raise AssertionError("ran a skipped test")

    @skipIf(True, "condition held")
    def test_skip_if(self) -> None:
        raise AssertionError("ran")

    @skipIf(False, "condition failed")
    def test_not_skipped(self) -> None:
        assert 1 + 1 == 2

    @expectedFailure
    def test_known_bug(self) -> None:
        assert 1 + 1 == 3, "arithmetic"

    @expectedFailure
    def test_fixed_bug(self) -> None:
        assert True

    def test_plain(self) -> None:
        assert 2 > 3, "two is not greater"


t = MyTest()
for name in sorted(n for n in dir(MyTest) if n.startswith("test")):
    print(t.run(name))
