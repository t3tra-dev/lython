# `itertools.chain(*rows)` flattens however many rows the sequence holds,
# and `itertools.product(*pools)` is every pick of one member from each
# pool -- one empty pick for no pools -- with written arguments before the
# star taking their places; `dict(**d)` is a new dict with d's items; and
# `range`, `max` and `min` take a starred sequence by its length, with the
# TypeError CPython raises for a length they refuse.
import itertools
import unittest


def flat(rows: list[list[str]]) -> str:
    return "".join(itertools.chain(*rows))


class StarredItertoolsTest(unittest.TestCase):
    def test_chain(self):
        rows = [[1, 2], [3]]
        self.assertEqual(list(itertools.chain(*rows)), [1, 2, 3])
        self.assertEqual(list(itertools.chain([0], *rows)), [0, 1, 2, 3])
        self.assertEqual(flat([["x"], [], ["y", "z"]]), "xyz")
        pairs = [(1, "a"), (2, "b")]
        self.assertEqual(list(itertools.chain(*pairs)), [1, "a", 2, "b"])
        total = 0
        for x in itertools.chain(*rows):
            total += x
        self.assertEqual(total, 6)

    def test_product(self):
        self.assertEqual(list(itertools.product(*[[1, 2], [3, 4], [5]])),
                         [(1, 3, 5), (1, 4, 5), (2, 3, 5), (2, 4, 5)])
        self.assertEqual(["".join(p) for p in itertools.product(*["ab", "cd"])],
                         ["ac", "ad", "bc", "bd"])
        self.assertEqual(list(itertools.product(*[range(2)] * 2)),
                         [(0, 0), (0, 1), (1, 0), (1, 1)])
        none: list[list[int]] = []
        self.assertEqual(list(itertools.product(*none)), [()])

    def test_a_dict_spread_into_dict(self):
        d = {"a": 1, "b": 2}
        copied = dict(**d)
        copied["z"] = 9
        self.assertEqual((copied, d), ({"a": 1, "b": 2, "z": 9},
                                       {"a": 1, "b": 2}))
        self.assertIsNot(dict(**d), d)

    def test_range_max_and_min_by_length(self):
        args = [2, 10, 3]
        self.assertEqual([list(range(*args)), list(range(*[4]))],
                         [[2, 5, 8], [0, 1, 2, 3]])
        xs = [3, 9, 2]
        self.assertEqual([max(*xs), min(*xs), max(*["abc"])], [9, 2, "c"])
        too_many = [1, 2, 3, 4]
        with self.assertRaises(TypeError) as raised:
            range(*too_many)
        self.assertEqual(str(raised.exception),
                         "range expected at most 3 arguments, got 4")
        nothing: list[int] = []
        with self.assertRaises(TypeError) as raised:
            max(*nothing)
        self.assertEqual(str(raised.exception),
                         "max expected at least 1 argument, got 0")
        one = [5]
        with self.assertRaises(TypeError) as raised:
            min(*one)
        self.assertEqual(str(raised.exception), "'int' object is not iterable")


if __name__ == "__main__":
    unittest.main()
