# A starred argument whose length only the run time knows fills the
# parameters it reaches, as CPython binds it: the defaults fill what a short
# sequence leaves, *args takes what a long one has past them, and any other
# length raises CPython's TypeError. The sequence is collected where CPython
# collects it: after a written positional, there; alone, after the keywords.
# Any iterable is collected so. `zip(*rows)` zips as many rows as it is given.
# A starred generator expression into a method or a format template is the
# tuple of its members, in a generator's body too, and `tuple`, `list` of one
# there have its members' type.
import os.path
import unittest


def add(a: int, b: int) -> int:
    return a + b


def tri(a: int, b: int = 10, c: int = 100) -> int:
    return a + b + c


def head(a: int, *rest: int) -> str:
    return f"{a}|{rest}"


def kw(a: int, b: int = 5, *, k: int = 0) -> int:
    return a + b + k


def collect(*a: int, key: int = 0) -> str:
    return f"{a}{key}"


class Shape:
    def __init__(self, w: int, h: int = 1) -> None:
        self.w = w
        self.h = h

    def scaled(self, by: int, plus: int = 0) -> int:
        return self.w * self.h * by + plus

    @staticmethod
    def diff(a: int, b: int) -> int:
        return a - b

    @classmethod
    def unit(cls, side: int) -> "Shape":
        return cls(side, side)


def summed(xs: list[int]):
    return add(*xs)


class Accumulator:
    def __init__(self, base: int) -> None:
        self.base = base

    def add(self, *values: int) -> int:
        return self.base + sum(values)

    def pair(self, a: int, b: int) -> int:
        return a * 10 + b


class StarredParametersTest(unittest.TestCase):
    def test_a_function(self):
        self.assertEqual(add(*[1, 2]), 3)
        self.assertEqual([tri(*list(range(1, n + 1))) for n in (1, 2, 3)],
                         [111, 103, 6])
        self.assertEqual(summed([3, 4]) + 1, 8)

    def test_the_lengths_cpython_refuses(self):
        none: list[int] = []
        with self.assertRaises(TypeError) as raised:
            add(*none)
        self.assertEqual(str(raised.exception),
                         "add() missing 2 required positional arguments: "
                         "'a' and 'b'")
        with self.assertRaises(TypeError) as raised:
            add(*[1])
        self.assertEqual(str(raised.exception),
                         "add() missing 1 required positional argument: 'b'")
        with self.assertRaises(TypeError) as raised:
            add(*[1, 2, 3])
        self.assertEqual(str(raised.exception),
                         "add() takes 2 positional arguments but 3 were given")
        with self.assertRaises(TypeError) as raised:
            tri(*[1, 2, 3, 4])
        self.assertEqual(str(raised.exception),
                         "tri() takes from 1 to 3 positional arguments but 4 "
                         "were given")
        with self.assertRaises(TypeError) as raised:
            kw(*[1, 2, 3], k=1)
        self.assertEqual(str(raised.exception),
                         "kw() takes from 1 to 2 positional arguments but 3 "
                         "positional arguments (and 1 keyword-only argument) "
                         "were given")

    def test_args_take_the_rest(self):
        self.assertEqual(head(*[1]), "1|()")
        self.assertEqual(head(*[1, 2, 3]), "1|(2, 3)")
        self.assertEqual(kw(*[1], k=100), 106)
        none: list[int] = []
        with self.assertRaises(TypeError) as raised:
            head(*none)
        self.assertEqual(str(raised.exception),
                         "head() missing 1 required positional argument: 'a'")

    def test_methods_and_constructors(self):
        s = Shape(*[2, 3])
        self.assertEqual((s.w, s.h), (2, 3))
        self.assertEqual(Shape(*[4]).h, 1)
        self.assertEqual(s.scaled(*[2]), 12)
        self.assertEqual(s.scaled(*[2, 1]), 13)
        self.assertEqual(Shape.diff(*[5, 1]), 4)
        self.assertEqual(s.diff(*[5, 1]), 4)
        self.assertEqual(Shape.unit(*[3]).h, 3)
        with self.assertRaises(TypeError) as raised:
            s.scaled(*[1, 2, 3])
        self.assertEqual(str(raised.exception),
                         "Shape.scaled() takes from 2 to 3 positional "
                         "arguments but 4 were given")
        none: list[int] = []
        with self.assertRaises(TypeError) as raised:
            Shape(*none)
        self.assertEqual(str(raised.exception),
                         "Shape.__init__() missing 1 required positional "
                         "argument: 'w'")

    def test_any_iterable(self):
        self.assertEqual(add(*range(2)), 1)
        self.assertEqual(add(*{4: "a", 3: "b"}), 7)
        self.assertEqual(add(*(5, 1)), 6)
        self.assertEqual(os.path.join(*["a", "b", "c"]), "a/b/c")

    def test_where_the_sequence_is_collected(self):
        alone = [1, 2, 3]
        self.assertEqual(collect(*alone, key=alone.pop()), "(1, 2)3")
        after = [1, 2, 3]
        self.assertEqual(collect(0, *after, key=after.pop()), "(0, 1, 2, 3)3")

    def test_zip_of_a_starred_sequence(self):
        rows = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(list(zip(*rows)), [(1, 4), (2, 5), (3, 6)])
        self.assertEqual([list(c) for c in zip(*[[1, 2], [3, 4], [5, 6]])],
                         [[1, 3, 5], [2, 4, 6]])
        none: list[list[int]] = []
        self.assertEqual(list(zip(*none)), [])
        self.assertEqual(list(zip(*[[1, 2, 3], [4]])), [(1, 4)])
        pairs = [(1, "a"), (2, "b")]
        texts: list[str] = []
        for column in zip(*pairs):
            for v in column:
                texts.append(v.upper() if isinstance(v, str) else str(v + 1))
        self.assertEqual(texts, ["2", "3", "A", "B"])


    def test_a_generator_expression_into_a_method_or_a_template(self):
        acc = Accumulator(10)
        words = ["x", "y"]
        self.assertEqual([acc.add(*(v for v in range(4))),
                          acc.add(1, *(v * 2 for v in range(3))),
                          acc.pair(*(v for v in (3, 4)))], [16, 17, 34])
        self.assertEqual("{} {}".format(*(w.upper() for w in words)), "X Y")


    def test_generator_expressions_in_a_generator_body(self):
        from typing import Iterator

        def gen(xs: list[int]) -> Iterator[int]:
            doubled = tuple(x * 2 for x in xs)
            yield doubled[0]
            yield list(x + 1 for x in xs)[1]
            yield Accumulator(1).add(*(x for x in xs))

        self.assertEqual(list(gen([4, 5])), [8, 6, 10])


if __name__ == "__main__":
    unittest.main()
