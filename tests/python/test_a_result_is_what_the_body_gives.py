# An unannotated function whose body can reach its end returns None there
# beside what its returns give. A generator annotated to yield float yields
# an int or a bool as itself, as a function annotated `-> float` returns
# one, and a float as a float.
import unittest
from typing import Generator, Iterator


def length_or_none(v: str | None):
    if v is not None:
        return len(v)


def sign_word(v: int):
    if v > 0:
        return "pos"
    elif v < 0:
        return "neg"


def first_over_two(xs: list[int]):
    for x in xs:
        if x > 2:
            return x


def doubled(v: int) -> Iterator[float]:
    yield v * 2


def mixed(v: int) -> Iterator[float]:
    yield v
    yield v / 2
    yield True


def counted(n: int) -> Generator[float, None, str]:
    for i in range(n):
        yield i
    return "done"


def present(xs: list[int | None]) -> Iterator[float]:
    for x in xs:
        if x is not None:
            yield x


class AFunctionThatCanReachItsEnd(unittest.TestCase):
    def test_none_beside_what_it_returns(self) -> None:
        self.assertEqual(length_or_none("abc"), 3)
        self.assertIsNone(length_or_none(None))
        self.assertEqual([sign_word(1), sign_word(-1), sign_word(0)],
                         ["pos", "neg", None])
        self.assertEqual(first_over_two([1, 3]), 3)
        self.assertIsNone(first_over_two([1]))


class AGeneratorAnnotatedToYieldFloat(unittest.TestCase):
    def test_an_int_is_yielded_as_an_int(self) -> None:
        self.assertEqual([type(v).__name__ for v in doubled(2)], ["int"])
        self.assertEqual(list(doubled(2)), [4])
        self.assertEqual(sum(doubled(5)), 10)

    def test_each_rung_as_itself(self) -> None:
        self.assertEqual(list(mixed(3)), [3, 1.5, True])
        self.assertEqual([type(v).__name__ for v in mixed(4)], ["int", "float", "bool"])
        self.assertEqual(list(present([1, None, 2])), [1, 2])

    def test_a_generator_that_returns_a_value(self) -> None:
        self.assertEqual(list(counted(3)), [0, 1, 2])
        self.assertEqual([x + 0.5 for x in counted(2)], [0.5, 1.5])


if __name__ == "__main__":
    unittest.main()
