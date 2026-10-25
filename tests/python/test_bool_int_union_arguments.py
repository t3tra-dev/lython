# A bool, or a value that is a bool or an int (a read of `{1: 5, 4: True}`),
# passed where a parameter declares int: through a Callable value, to a list's,
# set's or dict's method, to abs and round, to a method of the program's own
# class, to a nested function and to a function with a keyword-only parameter.
# The callee sees the value as it is: True stays a bool and adds as 1. (A call
# given such a value is bound to a local before it is compared: a tuple of them
# is typed before it is emitted, and that typing does not convert.)
import unittest
from typing import Callable


def kind(x: int) -> str:
    return type(x).__name__ + ":" + repr(x + 0)


def twice(x: int) -> int:
    return x * 2


def keyword(x: int, *, k: int = 0) -> str:
    return type(x).__name__ + ":" + repr(x + k)


def apply(f: Callable[[int], int], v: int) -> int:
    return f(v)


class Acc:
    def __init__(self) -> None:
        self.items: list[int] = []
        self.total = 0

    def put(self, v: int) -> None:
        self.items.append(v)
        self.total += v

    def kind(self, v: int) -> str:
        return type(v).__name__


class CallableValueTest(unittest.TestCase):

    def test_a_bool(self) -> None:
        f: Callable[[int], str] = kind
        g: Callable[[int], int] = twice
        self.assertEqual((f(True), f(7), g(True)), ("bool:1", "int:7", 2))

    def test_a_bool_or_an_int(self) -> None:
        d = {1: 5, 4: True}
        f: Callable[[int], str] = kind
        held = f(d[4])
        other = f(d[1])
        self.assertEqual((held, other), ("bool:1", "int:5"))
        doubled = apply(twice, d[4])
        self.assertEqual(doubled, 2)
        doubled = apply(twice, d[1])
        self.assertEqual(doubled, 10)


class ContainerMethodTest(unittest.TestCase):

    def test_append_add_and_store(self) -> None:
        d = {1: 5, 4: True}
        xs: list[int] = []
        xs.append(d[4])
        xs.append(d[1])
        s: set[int] = set()
        s.add(d[4])
        m: dict[str, int] = {}
        m["a"] = d[4]
        m["b"] = d[1]
        self.assertEqual(repr(xs), "[True, 5]")
        self.assertEqual([type(v).__name__ for v in xs], ["bool", "int"])
        self.assertEqual((repr(s), repr(m)), ("{True}", "{'a': True, 'b': 5}"))


class BuiltinTest(unittest.TestCase):

    def test_abs_and_round(self) -> None:
        d = {1: 5, 4: True}
        magnitude = abs(d[4])
        digits = round(2.567, d[1])
        self.assertEqual((repr(magnitude), type(magnitude).__name__),
                         ("1", "int"))
        self.assertEqual(repr(digits), "2.567")


class FunctionAndMethodTest(unittest.TestCase):

    def test_a_method(self) -> None:
        d = {1: 5, 4: True}
        a = Acc()
        a.put(d[4])
        a.put(d[1])
        self.assertEqual((repr(a.items), a.total), ("[True, 5]", 6))
        held = a.kind(d[4])
        other = a.kind(d[1])
        self.assertEqual((held, other), ("bool", "int"))
        method = a.kind
        self.assertEqual((method(True), method(3)), ("bool", "int"))

    def test_a_nested_function_and_a_keyword(self) -> None:
        d = {1: 5, 4: True}

        def inner(x: int) -> str:
            return type(x).__name__

        held = inner(d[4])
        other = inner(d[1])
        self.assertEqual((held, other), ("bool", "int"))
        held = keyword(d[4])
        other = keyword(d[1], k=1)
        self.assertEqual((held, other, keyword(True)),
                         ("bool:1", "int:6", "bool:1"))


if __name__ == "__main__":
    unittest.main()
