# A module function with no return annotation returns what a method of a class
# in the same module returns: called on a parameter, on a local, on a fresh
# instance, on a parameter typed only by its callers, and through iteration
# of the instance. One that returns from a `try` and from its handler
# returns the type both give, a literal among them included.
import unittest
from typing import Iterator


def parsed(s: str):
    try:
        return int(s)
    except ValueError:
        return -1


def named(s: str):
    try:
        return s.upper()
    except ValueError:
        return "none"


class Box:
    def __init__(self, xs: list[int]) -> None:
        self.xs = xs

    def size(self) -> int:
        return len(self.xs)

    def __iter__(self) -> Iterator[int]:
        return iter(self.xs)


def of_parameter(u: Box):
    return u.size()


def of_local():
    u = Box([1, 2, 3])
    return u.size()


def of_fresh_instance():
    return Box([4]).size()


def through_a_local(u: Box):
    r = u.size()
    return r + 1


def untyped(u):
    return u.size()


def calls_untyped():
    return untyped(Box([1, 2])) + 10


def counted(u: Box):
    return sum(1 for _ in u)


def largest(u: Box):
    return max(x for x in u)


class MethodResultTest(unittest.TestCase):
    def test_a_method_result_is_the_function_result(self):
        self.assertEqual(of_parameter(Box([1, 2])) + 1, 3)
        self.assertEqual(of_local() * 2, 6)
        self.assertEqual(of_fresh_instance() - 1, 0)
        self.assertEqual(through_a_local(Box([])) + 1, 2)

    def test_a_parameter_typed_by_its_callers(self):
        self.assertEqual(calls_untyped(), 12)

    def test_iterating_the_instance(self):
        self.assertEqual(counted(Box([7, 8, 9])) + 0, 3)
        self.assertEqual(largest(Box([7, 9, 8])) + 0, 9)


    def test_returns_from_a_try_and_its_handler(self):
        self.assertEqual([parsed("12"), parsed("x"), parsed("3") + 1],
                         [12, -1, 4])
        self.assertEqual(named("a"), "A")


if __name__ == "__main__":
    unittest.main()
