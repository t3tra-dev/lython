# A lambda handed to a generic function's `Callable[[T], R]`, or to a
# ParamSpec forwarder's `Callable[P, R]`, takes the parameter types the other
# arguments of the call give it, and its result is its own.
import unittest
from typing import Callable


def call[**P, R](fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    return fn(*args, **kwargs)


def maybe[**P, R](
    fn: Callable[P, R] | None, *args: P.args, **kwargs: P.kwargs
) -> R | None:
    if fn is None:
        return None
    return fn(*args, **kwargs)


def apply[T, R](fn: Callable[[T], R], x: T) -> R:
    return fn(x)


class LambdaArgumentTest(unittest.TestCase):

    def test_a_generic_callable_parameter(self) -> None:
        self.assertEqual(apply(lambda s: s.upper(), "q"), "Q")
        self.assertEqual(apply(lambda n: [n], 5), [5])
        self.assertEqual(apply(lambda n: n + 1, 2) << 1, 6)

    def test_a_param_spec_callable_parameter(self) -> None:
        self.assertEqual(call(lambda x: x + 1, 2) << 1, 6)
        self.assertEqual(call(lambda x, y: x * y, 3, y=4), 12)
        self.assertEqual(call(lambda *, sep: sep.join("ab"), sep="-"), "a-b")
        self.assertEqual(maybe(lambda s: s * 2, "ab"), "abab")

    def test_assert_raises_given_a_lambda(self) -> None:
        self.assertRaises(ValueError, lambda s: int(s), "x")
        with self.assertRaises(AssertionError):
            self.assertRaises(ValueError, lambda s: int(s), "7")


if __name__ == "__main__":
    unittest.main()
