# A parameter `Callable[P, R] | None` takes a function, a builtin, a class or
# a lambda, and the body calls it with the arguments the call passed on --
# in a function and in a method.
import unittest
from typing import Callable


def call_with[**P](f: Callable[P, object] | None = None, *args: P.args,
                   **kwargs: P.kwargs) -> object:
    if f is None:
        return "nothing to call"
    return f(*args, **kwargs)


class Runner:
    def run[**P](self, f: Callable[P, object] | None = None, *args: P.args,
                 **kwargs: P.kwargs) -> object:
        if f is None:
            return "nothing to run"
        return f(*args, **kwargs)


def add(a: int, b: int) -> int:
    return a + b


class OptionalParamSpecCallableTest(unittest.TestCase):

    def test_a_function(self) -> None:
        self.assertEqual(call_with(add, 1, b=2), 3)
        self.assertEqual(Runner().run(add, 2, 3), 5)

    def test_a_builtin(self) -> None:
        self.assertEqual(call_with(len, [1, 2]), 2)
        self.assertEqual(Runner().run(len, "abc"), 3)
        self.assertEqual(call_with(abs, -4), 4)

    def test_a_class(self) -> None:
        self.assertEqual(call_with(int, "7"), 7)
        self.assertEqual(Runner().run(str, 8), "8")

    def test_a_lambda_with_no_parameters(self) -> None:
        self.assertEqual(call_with(lambda: "lam"), "lam")
        self.assertEqual(Runner().run(lambda: "lam"), "lam")

    def test_nothing(self) -> None:
        self.assertEqual(call_with(), "nothing to call")
        self.assertEqual(Runner().run(), "nothing to run")


if __name__ == "__main__":
    unittest.main()
