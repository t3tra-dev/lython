# An int argument reaches a function that only reads it as an int -- in
# arithmetic, a comparison, a list, tuple or str index, and one return -- as
# the int it is: a literal, an operator's result, an int too wide for a
# 64-bit word, an element read from a list, and a bool where int is declared.
import unittest

WIDE = 2 ** 70


def scaled(x: int) -> int:
    return x * 3 + 1


def below(x: int, limit: int) -> bool:
    return x < limit


def pick(xs: list[int], i: int) -> int:
    return xs[i] * 10


def letter(s: str, i: int) -> str:
    return s[i]


def second(t: tuple[int, int, int], i: int) -> int:
    return t[i] + 1


def same(x: int) -> int:
    return x


class ArgumentTest(unittest.TestCase):

    def test_a_literal_and_an_operators_result(self) -> None:
        n = 4
        self.assertEqual((scaled(2), scaled(n * 5), scaled(-n)), (7, 61, -11))
        self.assertEqual((below(n + 1, 6), below(n + 2, 6)), (True, False))

    def test_an_int_wider_than_a_word(self) -> None:
        self.assertEqual(scaled(WIDE), WIDE * 3 + 1)
        self.assertEqual((below(WIDE, WIDE + 1), below(WIDE * 2, WIDE)),
                         (True, False))
        self.assertEqual((same(WIDE), same(WIDE * 2)), (WIDE, 2 ** 71))

    def test_an_element_read_from_a_list(self) -> None:
        xs = [5, WIDE, -7]
        self.assertEqual((scaled(xs[0]), scaled(xs[1]), scaled(xs[2])),
                         (16, WIDE * 3 + 1, -20))

    def test_an_index(self) -> None:
        xs = [1, 2, 3, 4]
        i = 2
        self.assertEqual((pick(xs, i), pick(xs, i - 3), pick(xs, i & 1)),
                         (30, 40, 10))
        self.assertEqual((letter("abcd", i + 1), second((7, 8, 9), i - 1)),
                         ("d", 9))
        with self.assertRaises(IndexError):
            pick(xs, i * 3)
        with self.assertRaises(IndexError):
            pick(xs, WIDE)

    def test_a_bool_where_int_is_declared(self) -> None:
        flags = [True, False]
        self.assertEqual((scaled(flags[0]), scaled(flags[1])), (4, 1))
        self.assertEqual(repr(same(flags[0])), "True")


if __name__ == "__main__":
    unittest.main()
