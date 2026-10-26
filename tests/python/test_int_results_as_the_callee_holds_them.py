# An int a function returns reaches its caller as the int it is -- a value
# known only as its i64, an operator's result, an int too wide for a 64-bit
# word, the caller's own argument handed back, an element of a list, and a
# bool where int is declared -- whether the caller does arithmetic with it,
# stores it, prints it, uses it as a key, joins it with another, or reaches
# the function through a callable value with several possible targets.
import unittest
from typing import Callable

WIDE = 2 ** 70


def plus(xs: list[int], x: int) -> int:
    return x + len(xs)


def first(xs: list[int]) -> int:
    return xs[0]


def same(x: int) -> int:
    return x


def square(x: int) -> int:
    return x * x


def seven(xs: list[int]) -> int:
    return 7


def as_int(flags: list[bool], i: int) -> int:
    return flags[i]


def pick_op(name: str) -> Callable[[int], int]:
    if name == "double":
        return lambda x: x * 2
    return lambda x: x + 100


class ResultTest(unittest.TestCase):

    def test_arithmetic_on_results(self) -> None:
        xs = [1, 2, 3]
        total = 0
        for i in range(5):
            total = total + plus(xs, i) * 2 + seven(xs)
        self.assertEqual(total, 85)
        self.assertEqual(plus(xs, WIDE) - WIDE, 3)

    def test_results_kept_as_objects(self) -> None:
        xs = [4, 5]
        kept = [plus(xs, 1), plus(xs, WIDE), first(xs), seven(xs)]
        self.assertEqual(kept, [3, WIDE + 2, 4, 7])
        table = {plus(xs, 10): "a", seven(xs): "b"}
        self.assertEqual((table[12], table[7]), ("a", "b"))
        self.assertEqual((repr(plus(xs, -9)), str(seven(xs))), ("-7", "7"))

    def test_an_argument_handed_back(self) -> None:
        n = 21
        self.assertEqual((same(n * 2), same(WIDE), same(-1)), (42, WIDE, -1))
        wides = [same(WIDE + k) for k in range(3)]
        self.assertEqual(wides, [WIDE, WIDE + 1, WIDE + 2])

    def test_a_square_too_wide_for_a_word(self) -> None:
        self.assertEqual((square(3), square(2 ** 40)), (9, 2 ** 80))

    def test_a_bool_where_int_is_declared(self) -> None:
        flags = [True, False]
        self.assertEqual((repr(as_int(flags, 0)), repr(as_int(flags, 1))),
                         ("True", "False"))

    def test_joined_and_dispatched(self) -> None:
        xs = [1]
        for flag in (True, False):
            joined = plus(xs, 5) if flag else seven(xs)
            self.assertEqual(joined, 6 if flag else 7)
        double = pick_op("double")
        shift = pick_op("shift")
        self.assertEqual((double(4), shift(4), double(WIDE)),
                         (8, 104, WIDE * 2))


if __name__ == "__main__":
    unittest.main()
