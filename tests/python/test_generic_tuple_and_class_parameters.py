# A type parameter is solved from every element of a tuple passed for
# `tuple[T, ...]`, and from every class passed for `type[E]`: elements of one
# type give that type, two unrelated ones their union, and two classes the
# nearest class both are.
import unittest


class A:
    pass


class B:
    pass


def first[T](t: tuple[T, ...]) -> T:
    return t[0]


def count[T](t: tuple[T, ...]) -> int:
    return len(t)


def names[E: BaseException](t: tuple[type[E], ...]) -> list[str]:
    return [c.__name__ for c in t]


def both[E](a: type[E], b: type[E]) -> str:
    return a.__name__ + b.__name__


class SolveTest(unittest.TestCase):

    def test_a_tuple_of_values(self) -> None:
        self.assertEqual((first((1, 2, 3)), first(("a", "b"))), (1, "a"))
        self.assertEqual(count((1, "a", 2.5)), 3)

    def test_a_tuple_of_classes(self) -> None:
        self.assertEqual(names((KeyError, IndexError, LookupError)),
                         ["KeyError", "IndexError", "LookupError"])

    def test_two_class_arguments(self) -> None:
        self.assertEqual((both(A, B), both(KeyError, IndexError)),
                         ("AB", "KeyErrorIndexError"))


if __name__ == "__main__":
    unittest.main()
