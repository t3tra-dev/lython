# `issubclass` and `isinstance` given a tuple of classes whose length only the
# run knows, or a union of a class and such a tuple, test against any of them.
import unittest


class Expectation:
    def __init__(self, expected: type[Exception] | tuple[type[Exception], ...]) -> None:
        self.expected = expected

    def matches(self, e: Exception) -> bool:
        return isinstance(e, self.expected)

    def covers(self, t: type[Exception]) -> bool:
        return issubclass(t, self.expected)


def subclass_of_any(t: type[Exception], classes: tuple[type[Exception], ...]) -> bool:
    return issubclass(t, classes)


def instance_of_any(x: object, classes: tuple[type[Exception], ...]) -> bool:
    return isinstance(x, classes)


class TupleOfClassesTest(unittest.TestCase):

    def test_a_tuple(self) -> None:
        self.assertTrue(subclass_of_any(KeyError, (ValueError, LookupError)))
        self.assertFalse(subclass_of_any(KeyError, (ValueError, TypeError)))
        self.assertFalse(subclass_of_any(KeyError, ()))
        self.assertTrue(instance_of_any(ValueError(), (KeyError, ValueError)))
        self.assertFalse(instance_of_any(3, ()))

    def test_a_class_or_a_tuple(self) -> None:
        self.assertTrue(Expectation(KeyError).matches(KeyError()))
        self.assertTrue(Expectation((ValueError, KeyError)).matches(KeyError()))
        self.assertFalse(Expectation((ValueError,)).matches(KeyError()))
        self.assertTrue(Expectation(LookupError).covers(IndexError))
        self.assertTrue(Expectation((TypeError, LookupError)).covers(IndexError))
        self.assertFalse(Expectation(()).covers(KeyError))


if __name__ == "__main__":
    unittest.main()
