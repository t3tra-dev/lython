# The class of a class value is `type`, whichever class it names and however
# the value was reached.
import sys
import unittest


class Point:
    pass


class ClassOfAClassTest(unittest.TestCase):

    def test_builtin_and_source_classes(self) -> None:
        self.assertIs(type(KeyError), type)
        self.assertIs(type(Point), type)
        self.assertEqual(type(int).__name__, "type")

    def test_a_class_value_in_a_variable(self) -> None:
        cls: type[LookupError] = KeyError if len(sys.argv) > 0 else IndexError
        self.assertIs(type(cls), type)
        self.assertIs(type(type(cls)), type)

    def test_a_union_holding_a_class(self) -> None:
        value: type[KeyError] | int = KeyError if len(sys.argv) > 0 else 3
        self.assertEqual(repr(type(value)), "<class 'type'>")


if __name__ == "__main__":
    unittest.main()
