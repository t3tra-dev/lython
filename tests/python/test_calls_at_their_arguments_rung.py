# A call whose argument is a lower numeric rung than its parameter -- True
# where int is declared, 3 where float is -- runs the function on the argument
# as it is, and the call's value has that reading's type wherever the call is
# written: returned from a function annotated with the higher rung, as an arm
# of a conditional, given to repr, from inside another such function, and from
# a function written above the one it calls. A decorated function is reached
# through its decorator.
import unittest
from typing import Callable


def returns_flag() -> int:
    return echo(True)


def returns_count() -> float:
    return kept(3)


def relay(b: int) -> int:
    return echo(b)


def doubled(x: float) -> float:
    return scaled(x)


def echo(x: int) -> int:
    return x


def kept(x: float) -> float:
    return x


def scaled(x: float) -> float:
    return x * 2


calls: list[str] = []


def logged(fn: Callable[[int], int]) -> Callable[[int], int]:
    def wrapper(x: int) -> int:
        calls.append("wrapped")
        return fn(x)
    return wrapper


@logged
def logged_echo(x: int) -> int:
    return x


class ReturnTest(unittest.TestCase):

    def test_a_bool_returned_under_int(self) -> None:
        self.assertEqual((repr(returns_flag()), type(returns_flag()).__name__),
                         ("True", "bool"))

    def test_an_int_returned_under_float(self) -> None:
        self.assertEqual((repr(returns_count()), type(returns_count()).__name__),
                         ("3", "int"))


class ExpressionTest(unittest.TestCase):

    def test_an_arm_of_a_conditional(self) -> None:
        on = len(calls) >= 0
        self.assertEqual(repr(echo(True) if on else 5), "True")
        self.assertEqual(repr(5 if on else echo(True)), "5")

    def test_repr_and_str(self) -> None:
        self.assertEqual((repr(echo(True)), str(echo(False)), repr(kept(3))),
                         ("True", "False", "3"))


class ChainTest(unittest.TestCase):

    def test_a_function_calling_another(self) -> None:
        self.assertEqual((repr(relay(True)), type(relay(True)).__name__),
                         ("True", "bool"))
        self.assertEqual((repr(doubled(4)), repr(doubled(2.5))), ("8", "5.0"))


class DecoratorTest(unittest.TestCase):

    def test_the_decorator_runs(self) -> None:
        calls.clear()
        self.assertEqual(repr(logged_echo(True)), "True")
        self.assertEqual(repr(logged_echo(7)), "7")
        self.assertEqual(calls, ["wrapped", "wrapped"])


if __name__ == "__main__":
    unittest.main()
