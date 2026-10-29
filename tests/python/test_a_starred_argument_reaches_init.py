# A starred argument whose members the type or the spelling says reaches the
# parameters one by one -- of a function, a method, or a class's `__init__`,
# which runs; a starred tuple display is its elements; a starred generator
# expression is its members, its body running where CPython iterates it; and
# positionals beside a `**` mapping stay the callee's *args.
import unittest
from dataclasses import dataclass
from typing import NamedTuple


class Scaled:
    def __init__(self, a: int, b: int) -> None:
        self.a = a * 10
        self.b = b * 10


class Summed:
    def __init__(self, a: int, b: int, c: int = 9) -> None:
        self.s = a + b + c


class Row(NamedTuple):
    a: int
    b: str
    c: float = 0.5


@dataclass
class Pair:
    a: int
    b: int = 7


class Forwarded:
    def __init__(self, *parts: str, **options: int) -> None:
        self.parts = parts
        self.options = options


def both(a: float, b: float = 0.5) -> str:
    return f"{a!r} {b!r}"


def first[T](a: T, b: T) -> T:
    return a


def keyed(a: int, b: int = 0, *, k: int = 0) -> str:
    return f"{a},{b},{k}"


class StarredInitTest(unittest.TestCase):
    def test_init_runs_for_a_starred_tuple(self):
        t = (1, 2)
        made = Scaled(*t)
        self.assertEqual((made.a, made.b), (10, 20))
        literal = Scaled(*(3, 4))
        self.assertEqual((literal.a, literal.b), (30, 40))

    def test_defaults_fill_what_the_tuple_leaves(self):
        t = (1, 2)
        self.assertEqual([Summed(*t).s, Summed(*(1, 2)).s, Summed(1, *(2,)).s],
                         [12, 12, 12])
        self.assertEqual(repr(Row(*(1, "x"))), "Row(a=1, b='x', c=0.5)")
        self.assertEqual(repr(Pair(*(3, 4))), "Pair(a=3, b=4)")

    def test_positionals_beside_a_mapping_are_the_args(self):
        options = {"k": 1}
        made = Forwarded("a", "b", **options)
        self.assertEqual((made.parts, made.options), (("a", "b"), {"k": 1}))

    def test_a_tuple_display_meets_each_parameter(self):
        self.assertEqual([both(*(4, 5)), both(*(4.0, 5)), both(4, *(5,))],
                         ["4 5", "4.0 5", "4 5"])
        self.assertEqual(first(*(1.5, 2.5)), 1.5)

    def test_a_generator_expression(self):
        self.assertEqual(keyed(*(x * 2 for x in range(2))), "0,2,0")
        self.assertEqual(keyed(1, *(x for x in range(1))), "1,0,0")
        numbers = iter([10, 20, 30])
        self.assertEqual(
            keyed(*(next(numbers) for _ in range(2)), k=next(numbers)),
            "20,30,10")
        words = [1, 2]
        self.assertEqual(keyed(*(w for w in words), k=words.pop()), "1,0,2")


if __name__ == "__main__":
    unittest.main()
