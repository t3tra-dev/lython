# A function whose return annotation is int or float returns what its body
# returns: a module-level name bound to a bool is returned as the bool, whether
# the name is bound above the function or below it, and so is a function that
# returns such a function's result. A name whose value is computed from another
# module-level name has that value's type.
import unittest

ON = True
RATE = 2.5
TABLE = {1: 5, 4: True}
PICKED = TABLE[4]
SIZES = [3, 4]
COUNT = len(SIZES)


def as_int() -> int:
    return ON


def as_float() -> float:
    return ON


def pick(c: bool) -> float:
    if c:
        return ON
    return RATE


def outer() -> int:
    return inner()


def inner() -> int:
    return LATER


LATER = False


def picked() -> int:
    return PICKED


def doubled() -> int:
    return COUNT * 2


class ReturnTest(unittest.TestCase):

    def test_a_bool_name_under_int(self) -> None:
        self.assertEqual((repr(as_int()), type(as_int()).__name__, as_int() + 1),
                         ("True", "bool", 2))

    def test_a_bool_name_under_float(self) -> None:
        self.assertEqual((repr(as_float()), type(as_float()).__name__),
                         ("True", "bool"))

    def test_arms_of_two_rungs(self) -> None:
        self.assertEqual((repr(pick(True)), repr(pick(False))), ("True", "2.5"))

    def test_a_name_bound_below(self) -> None:
        self.assertEqual((repr(inner()), repr(outer()),
                          type(outer()).__name__), ("False", "False", "bool"))

    def test_a_name_computed_from_another(self) -> None:
        self.assertEqual((repr(picked()), type(picked()).__name__),
                         ("True", "bool"))
        self.assertEqual(doubled(), 4)


if __name__ == "__main__":
    unittest.main()
