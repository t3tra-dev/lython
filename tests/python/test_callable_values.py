# WHAT: builtins and classes used as Callable values -- passed to a Callable
# parameter, stored in a Callable-typed name or list, returned, mapped, and
# handed through a ParamSpec forwarder (a function's and unittest's own
# addCleanup) -- with the results CPython 3.14 gives.
import unittest
from collections.abc import Callable


class Point:
    def __init__(self, x: int, y: int = 0) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return 'Point(%d, %d)' % (self.x, self.y)


def apply1[A, R](f: Callable[[A], R], value: A) -> R:
    return f(value)


def apply_str(f: Callable[[str], int], value: str) -> int:
    return f(value)


def call_with[**P](fn: Callable[P, object], *args: P.args, **kwargs: P.kwargs) -> object:
    return fn(*args, **kwargs)


def make_point_factory() -> Callable[[int], Point]:
    return Point


cleanup_log: list[str] = []


class CleanupProbe(unittest.TestCase):
    def test_it(self) -> None:
        self.addCleanup(cleanup_log.append, 'appended')
        self.addCleanup(Point, 1)


class ClassesAsCallables(unittest.TestCase):
    def test_passed_to_a_callable_parameter(self) -> None:
        self.assertEqual(apply_str(int, '12'), 12)
        self.assertEqual(apply_str(len, 'abc'), 3)
        self.assertEqual(repr(apply1(Point, 3)), 'Point(3, 0)')

    def test_stored_returned_and_listed(self) -> None:
        make: Callable[[int, int], Point] = Point
        self.assertEqual(repr(make(1, 2)), 'Point(1, 2)')
        self.assertEqual(repr(make_point_factory()(7)), 'Point(7, 0)')
        parsers: list[Callable[[str], int]] = [int, len]
        self.assertEqual([p('42') for p in parsers], [42, 2])

    def test_mapped(self) -> None:
        self.assertEqual(repr(list(map(Point, [1, 2]))), '[Point(1, 0), Point(2, 0)]')
        self.assertEqual(list(map(str, [1, 2])), ['1', '2'])


def to_str(f: Callable[[str], str], value: str) -> str:
    return f(value)


def reduce_ints(f: Callable[[list[int]], int], values: list[int]) -> int:
    return f(values)


def round_float(f: Callable[[float], int], value: float) -> int:
    return f(value)


class BuiltinsAsCallables(unittest.TestCase):
    def test_builtins_answered_at_their_call(self) -> None:
        self.assertEqual(to_str(repr, 'x'), "'x'")
        self.assertEqual(reduce_ints(min, [3, 1, 2]), 1)
        self.assertEqual(reduce_ints(max, [3, 1, 2]), 3)
        self.assertEqual(reduce_ints(sum, [3, 1, 2]), 6)
        self.assertEqual(round_float(round, 2.5), 2)

    def test_builtins_with_a_function_object(self) -> None:
        self.assertEqual(apply_str(len, 'abc'), 3)
        self.assertEqual(apply1(chr, 65), 'A')


class ThroughParamSpec(unittest.TestCase):
    def test_function_forwarder(self) -> None:
        self.assertEqual(call_with(int, '42'), 42)
        self.assertEqual(call_with(int, 'ff', 16), 255)
        self.assertEqual(repr(call_with(divmod, 7, 2)), '(3, 1)')
        self.assertEqual(repr(call_with(Point, 1, 2)), 'Point(1, 2)')
        with self.assertRaises(ValueError):
            call_with(int, 'x')

    def test_add_cleanup(self) -> None:
        cleanup_log.clear()
        result = unittest.TestResult()
        CleanupProbe('test_it').run(result)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(cleanup_log, ['appended'])

if __name__ == '__main__':
    unittest.main()
