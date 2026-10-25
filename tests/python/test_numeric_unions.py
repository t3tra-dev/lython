# A value whose type is a union of numeric rungs -- `bool | int` out of
# `{1: 5, 4: True}` or `[5, True]`, `int | float` out of `3 if c else 2.5` --
# crossing a boundary that declares the higher rung stays the value it is:
# passed to a parameter, bound to an annotated local, or returned from a
# function whose arms return different rungs. A bool still prints True and is
# a bool inside the callee; an int still prints without a fraction.
import unittest


def twice(x: int) -> int:
    return x * 2


def shown(x: int) -> str:
    return repr(x) + ":" + type(x).__name__


def scaled(x: float) -> float:
    return x * 2.0


def pick(c: bool) -> int:
    return 7 if c else True


def halved(n: int) -> float:
    return n / 2 if n % 2 != 0 else n // 2


class ParameterTest(unittest.TestCase):

    def test_a_dict_value_of_bool_or_int(self) -> None:
        d = {1: 5, 4: True}
        self.assertEqual((twice(d[1]), twice(d[4])), (10, 2))
        self.assertEqual((shown(d[1]), shown(d[4])), ("5:int", "True:bool"))

    def test_in_a_loop(self) -> None:
        d = {1: 5, 4: True, 7: False}
        total = 0
        names: list[str] = []
        for k in [1, 4, 7]:
            total += twice(d[k])
            names.append(shown(d[k]))
        self.assertEqual(total, 12)
        self.assertEqual(names, ["5:int", "True:bool", "False:bool"])

    def test_a_list_literal_element(self) -> None:
        xs = [5, True]
        self.assertEqual((twice(xs[0]), shown(xs[1])), (10, "True:bool"))

    def test_an_int_or_a_float(self) -> None:
        for c in [True, False]:
            u = 3 if c else 2.5
            self.assertEqual(repr(scaled(u)), "6.0" if c else "5.0")


class LocalTest(unittest.TestCase):

    def test_an_annotated_local(self) -> None:
        d = {1: 5, 4: True}
        x: int = d[4]
        self.assertEqual((repr(x), x + 1), ("True", 2))
        for c in [True, False]:
            u = 3 if c else 2.5
            v: float = u
            self.assertEqual(repr(v), "3" if c else "2.5")


class ReturnTest(unittest.TestCase):

    def test_arms_of_different_rungs(self) -> None:
        self.assertEqual((repr(pick(True)), repr(pick(False))), ("7", "True"))
        self.assertEqual(twice(pick(False)), 2)
        self.assertEqual((repr(halved(7)), repr(halved(8))), ("3.5", "4"))


if __name__ == "__main__":
    unittest.main()
