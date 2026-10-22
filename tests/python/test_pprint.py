# WHAT: pprint.pformat's layout of lists, tuples, dicts, sets, frozensets,
# strs and bytes past the width, with indent, width, depth, compact,
# sort_dicts and underscore_numbers, plus saferepr / isreadable /
# isrecursive, compared with what CPython 3.14's pprint returns.
import pprint
import unittest


class Pprint(unittest.TestCase):
    def test_short_list(self) -> None:
        self.assertEqual(
            pprint.pformat([1, 2, 3]),
            '[1, 2, 3]')

    def test_sorted_dict(self) -> None:
        self.assertEqual(
            pprint.pformat({'b': 1, 'a': [1, 2]}),
            "{'a': [1, 2], 'b': 1}")

    def test_long_list(self) -> None:
        self.assertEqual(
            pprint.pformat(list(range(30))),
            '[0,\n 1,\n 2,\n 3,\n 4,\n 5,\n 6,\n 7,\n 8,\n 9,\n 10,\n 11,\n 12,\n 13,\n 14,\n 15,\n 16,\n 17,\n 18,\n 19,\n 20,\n 21,\n 22,\n 23,\n 24,\n 25,\n 26,\n 27,\n 28,\n 29]')

    def test_dict_of_lists(self) -> None:
        self.assertEqual(
            pprint.pformat({'key%d' % i: list(range(i)) for i in range(8)}),
            "{'key0': [],\n 'key1': [0],\n 'key2': [0, 1],\n 'key3': [0, 1, 2],\n 'key4': [0, 1, 2, 3],\n 'key5': [0, 1, 2, 3, 4],\n 'key6': [0, 1, 2, 3, 4, 5],\n 'key7': [0, 1, 2, 3, 4, 5, 6]}")

    def test_tuples(self) -> None:
        self.assertEqual(
            pprint.pformat([(1, 'a'), (2, 'b')] * 8),
            "[(1, 'a'),\n (2, 'b'),\n (1, 'a'),\n (2, 'b'),\n (1, 'a'),\n (2, 'b'),\n (1, 'a'),\n (2, 'b'),\n (1, 'a'),\n (2, 'b'),\n (1, 'a'),\n (2, 'b'),\n (1, 'a'),\n (2, 'b'),\n (1, 'a'),\n (2, 'b')]")

    def test_set(self) -> None:
        self.assertEqual(
            pprint.pformat({3, 1, 2}),
            '{1, 2, 3}')

    def test_frozenset(self) -> None:
        self.assertEqual(
            pprint.pformat(frozenset({3, 1, 2})),
            'frozenset({1, 2, 3})')

    def test_long_set(self) -> None:
        self.assertEqual(
            pprint.pformat(set(range(25))),
            '{0,\n 1,\n 2,\n 3,\n 4,\n 5,\n 6,\n 7,\n 8,\n 9,\n 10,\n 11,\n 12,\n 13,\n 14,\n 15,\n 16,\n 17,\n 18,\n 19,\n 20,\n 21,\n 22,\n 23,\n 24}')

    def test_long_str(self) -> None:
        self.assertEqual(
            pprint.pformat('word ' * 30),
            "('word word word word word word word word word word word word word word word '\n 'word word word word word word word word word word word word word word word ')")

    def test_str_in_list(self) -> None:
        self.assertEqual(
            pprint.pformat(['word ' * 30]),
            "['word word word word word word word word word word word word word word word '\n 'word word word word word word word word word word word word word word word ']")

    def test_bytes(self) -> None:
        self.assertEqual(
            pprint.pformat(b'abcdefgh' * 15),
            "(b'abcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcd'\n b'efghabcdefghabcdefghabcdefghabcdefghabcdefgh')")

    def test_compact(self) -> None:
        self.assertEqual(
            pprint.pformat(list(range(40)), compact=True),
            '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,\n 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]')

    def test_indent_width(self) -> None:
        self.assertEqual(
            pprint.pformat(list(range(10)), width=20, indent=4),
            '[   0,\n    1,\n    2,\n    3,\n    4,\n    5,\n    6,\n    7,\n    8,\n    9]')

    def test_depth(self) -> None:
        self.assertEqual(
            pprint.pformat({'x': {'y': {'z': [1, 2]}}}, depth=2),
            "{'x': {'y': {...}}}")

    def test_underscores(self) -> None:
        self.assertEqual(
            pprint.pformat(1234567, underscore_numbers=True),
            '1_234_567')

    def test_unsorted(self) -> None:
        self.assertEqual(
            pprint.pformat({'b': 1, 'a': 2}, sort_dicts=False),
            "{'b': 1, 'a': 2}")

    def test_saferepr(self) -> None:
        self.assertEqual(
            pprint.saferepr({'a': [1, (2, 3)]}),
            "{'a': [1, (2, 3)]}")

    def test_flags(self) -> None:
        self.assertEqual(
            (pprint.isreadable([1, 2]), pprint.isrecursive([1, 2])),
            (True, False))

    def test_empties(self) -> None:
        self.assertEqual(
            (pprint.pformat((1,)), pprint.pformat(()), pprint.pformat([]), pprint.pformat({}), pprint.pformat(set())),
            ('(1,)', '()', '[]', '{}', 'set()'))

    def test_scalars(self) -> None:
        self.assertEqual(
            pprint.pformat([1.5, None, True, 's']),
            "[1.5, None, True, 's']")


if __name__ == "__main__":
    unittest.main()
