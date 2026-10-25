# Ordering answers what CPython's does on every path a sort or a comparison
# can take: a list whose slots are all immediate ints, all exact strs or all
# exact tuples (list.sort chooses its comparison once, as CPython's listsort
# does), a list that mixes immediates with objects, and the comparisons that
# max, min and tuple ordering reach without a sort.
import unittest


def insertion_sorted(xs: list[int]) -> list[int]:
    out: list[int] = []
    for x in xs:
        i = len(out)
        out.append(x)
        while i > 0 and out[i - 1] > x:
            out[i] = out[i - 1]
            i -= 1
        out[i] = x
    return out


class SortKindTest(unittest.TestCase):

    def test_literal_ints(self) -> None:
        xs = [5, -3, 0, 12, -3, 7, -100, 1]
        xs.sort()
        self.assertEqual(xs, [-100, -3, -3, 0, 1, 5, 7, 12])

    def test_computed_ints(self) -> None:
        # Computed, so each slot holds the value itself rather than a shared
        # literal object; long enough to merge blocks.
        xs = [(i * 7919) % 1009 - 500 for i in range(300)]
        expected = insertion_sorted(xs)
        xs.sort()
        self.assertEqual(xs, expected)

    def test_computed_ints_with_one_past_the_immediate_range(self) -> None:
        xs = [(i * 7919) % 1009 - 500 for i in range(100)]
        xs.append(2 ** 62 + len(xs))
        xs.append(-(2 ** 62) - len(xs))
        expected = insertion_sorted(xs)
        xs.sort()
        self.assertEqual(xs, expected)

    def test_ints_past_the_immediate_range(self) -> None:
        big = 2 ** 62
        xs = [big, -big - 1, 3, 2 ** 100, -(2 ** 100), big - 1, -big]
        xs.sort()
        self.assertEqual(xs, [-(2 ** 100), -big - 1, -big, 3, big - 1, big,
                              2 ** 100])

    def test_bools_among_ints(self) -> None:
        xs = [True, 0, -1, False, 1]
        xs.sort()
        self.assertEqual(repr(xs), "[-1, 0, False, True, 1]")

    def test_strs_of_every_width(self) -> None:
        xs = ["b", "aĀ", "a", "a\xe9", "\U0001f600", "", "ab", "a\U0001f600"]
        xs.sort()
        self.assertEqual(xs, ["", "a", "ab", "a\xe9", "aĀ", "a\U0001f600",
                              "b", "\U0001f600"])

    def test_strs_sharing_a_prefix(self) -> None:
        xs = ["w10", "w1", "w100", "w09", "w"]
        xs.sort()
        self.assertEqual(xs, ["w", "w09", "w1", "w10", "w100"])

    def test_tuples_are_stable_on_equal_keys(self) -> None:
        xs = [(1, "b"), (0, "z"), (1, "a"), (0, "y")]
        xs.sort()
        self.assertEqual(xs, [(0, "y"), (0, "z"), (1, "a"), (1, "b")])

    def test_tuples_of_different_lengths(self) -> None:
        xs = [(1, 2, 3), (1, 2), (1,), (0, 9)]
        xs.sort()
        self.assertEqual(repr(xs), "[(0, 9), (1,), (1, 2), (1, 2, 3)]")

    def test_one_nan_object_is_equal_to_itself_inside_a_tuple(self) -> None:
        nan = float("nan")
        xs = [(nan, 1), (nan, 0)]
        xs.sort()
        self.assertEqual([t[1] for t in xs], [0, 1])

    def test_two_nan_objects_leave_the_order_alone(self) -> None:
        xs = [(float("nan"), 1), (float("nan"), 0)]
        xs.sort()
        self.assertEqual([t[1] for t in xs], [1, 0])

    def test_floats_keep_equal_zeros_in_order(self) -> None:
        xs = [2.5, 0.0, -1.0, -0.0, 1e300]
        xs.sort()
        self.assertEqual(repr(xs), "[-1.0, 0.0, -0.0, 2.5, 1e+300]")

    def test_key_sort_is_stable_both_ways(self) -> None:
        words = ["bb", "a", "cc", "d", "ee"]
        self.assertEqual(sorted(words, key=len), ["a", "d", "bb", "cc", "ee"])
        self.assertEqual(sorted(words, key=len, reverse=True),
                         ["bb", "cc", "ee", "a", "d"])


def stable_sorted_by(xs: list[tuple[int, int]], descending: bool) -> list[tuple[int, int]]:
    # Insertion sort on the second field, equal keys kept in input order.
    out: list[tuple[int, int]] = []
    for x in xs:
        i = len(out)
        out.append(x)
        while i > 0 and ((out[i - 1][1] < x[1]) if descending
                         else (out[i - 1][1] > x[1])):
            out[i] = out[i - 1]
            i -= 1
        out[i] = x
    return out


def second(t: tuple[int, int]) -> int:
    return t[1]


calls_seen: list[int] = []


def picked(store: list[list[int]]) -> list[int]:
    calls_seen.append(1)
    return store[len(calls_seen) - 1]


class KeySortTest(unittest.TestCase):

    def test_elements_move_with_their_keys_through_the_merges(self) -> None:
        # Long enough to merge blocks; few distinct keys, so stability shows.
        xs = [(i, (i * 7919) % 37) for i in range(500)]
        self.assertEqual(sorted(xs, key=second),
                         stable_sorted_by(xs, False))

    def test_reverse_keeps_equal_keys_in_input_order(self) -> None:
        xs = [(i, (i * 7919) % 37) for i in range(500)]
        self.assertEqual(sorted(xs, key=second, reverse=True),
                         stable_sorted_by(xs, True))

    def test_list_sort_with_a_key_sorts_the_list_itself(self) -> None:
        xs = [(i, (i * 31) % 7) for i in range(100)]
        alias = xs
        xs.sort(key=lambda t: t[1], reverse=True)
        self.assertTrue(alias is xs)
        self.assertEqual(alias, stable_sorted_by(
            [(i, (i * 31) % 7) for i in range(100)], True))

    def test_the_receiver_is_evaluated_once(self) -> None:
        calls_seen.clear()
        store = [[3, 1, 2], [9, 8]]
        picked(store).sort(key=lambda v: -v)
        self.assertEqual(len(calls_seen), 1)
        self.assertEqual(store, [[3, 2, 1], [9, 8]])

    def test_str_keys(self) -> None:
        words = ["b", "A", "c", "B", "a"]
        self.assertEqual(sorted(words, key=str.lower),
                         ["A", "a", "b", "B", "c"])

    def test_a_raising_key_leaves_the_list_alone(self) -> None:
        xs = [3, 1, 2]

        def key(v: int) -> int:
            if v == 2:
                raise ValueError("no")
            return v

        with self.assertRaises(ValueError):
            xs.sort(key=key)
        self.assertEqual(xs, [3, 1, 2])

    def test_unorderable_keys(self) -> None:
        with self.assertRaises(TypeError) as caught:
            sorted([1, 2], key=lambda v: v if v == 1 else "s")
        self.assertEqual(str(caught.exception),
                         "'<' not supported between instances of 'str' and "
                         "'int'")

    def test_a_key_that_grows_the_list(self) -> None:
        xs = [3, 1, 2]

        def key(v: int) -> int:
            if len(xs) < 5:
                xs.append(0)
            return v

        with self.assertRaises(ValueError) as caught:
            xs.sort(key=key)
        self.assertEqual(str(caught.exception), "list modified during sort")


class StrSortTest(unittest.TestCase):

    def test_strs_sharing_prefixes_through_the_merges(self) -> None:
        xs = ["w" + str((i * 7919) % 1009) for i in range(600)]
        expected = sorted(xs, key=lambda s: [ord(c) for c in s])
        xs.sort()
        self.assertEqual(xs, expected)

    def test_concatenation_results_stay_appendable(self) -> None:
        s = "w" + str(len("abc"))
        t = s
        s += "x"
        s += "y"
        self.assertEqual((s, t), ("w3xy", "w3"))


class OrderWithoutSortTest(unittest.TestCase):

    def test_max_and_min_of_computed_ints(self) -> None:
        xs = [(i * 37) % 101 - 50 for i in range(101)]
        self.assertEqual(max(xs), 50)
        self.assertEqual(min(xs), -50)

    def test_max_and_min_of_computed_floats(self) -> None:
        xs = [((i * 37) % 101 - 50) / 4 for i in range(101)]
        self.assertEqual(max(xs), 12.5)
        self.assertEqual(min(xs), -12.5)

    def test_tuples_of_computed_ints(self) -> None:
        a = (len("ab"), len("abc") * 10)
        b = (len("ab"), len("abcd") * 10)
        self.assertTrue(a < b)
        self.assertFalse(b <= a)
        self.assertTrue(a == (2, 30))

    def test_max_and_min_of_ints(self) -> None:
        xs = [3, -5, 2 ** 63, 7]
        self.assertEqual(max(xs), 2 ** 63)
        self.assertEqual(min(xs), -5)

    def test_min_and_max_return_the_first_of_equal_floats(self) -> None:
        self.assertEqual(repr(min([1.5, -0.0, 0.0])), "-0.0")
        self.assertEqual(repr(max([0.0, -0.0])), "0.0")

    def test_max_and_min_of_strs(self) -> None:
        words = ["b", "ab", "c", "ca", "\U0001f600", ""]
        self.assertEqual(max(words), "\U0001f600")
        self.assertEqual(min(words), "")
        self.assertEqual(max(["b", "ab", "c"]), "c")

    def test_nan_is_kept_when_nothing_compares_past_it(self) -> None:
        nan = float("nan")
        first = max([nan, 1.0, 2.0])
        self.assertTrue(first != first)
        self.assertEqual(min([1.0, nan, 0.5]), 0.5)
        self.assertEqual(max([1.0, nan]), 1.0)

    def test_the_first_of_equal_ints_is_answered(self) -> None:
        self.assertEqual(repr(max([len("a"), True])), "1")
        self.assertEqual(repr(min([True, len("a")])), "True")
        self.assertEqual(repr(max([False, True, len("a")])), "True")

    def test_a_temporary_list(self) -> None:
        self.assertEqual(max([x * 3 % 11 for x in range(20)]), 10)
        self.assertEqual(min([x * 3 % 11 for x in range(1, 20)]), 0)

    def test_an_empty_list_raises(self) -> None:
        xs = [len("ab")]
        with self.assertRaises(ValueError) as caught:
            max(xs[:0])
        self.assertEqual(str(caught.exception),
                         "max() iterable argument is empty")
        with self.assertRaises(ValueError) as caught:
            min(xs[:0])
        self.assertEqual(str(caught.exception),
                         "min() iterable argument is empty")

    def test_tuple_operators(self) -> None:
        a = (1, 2, 3)
        b = (1, 2, 4)
        self.assertTrue(a < b)
        self.assertTrue(a <= b)
        self.assertFalse(a > b)
        self.assertFalse(a >= b)
        self.assertTrue((1, 2) < (1, 2, 0))
        self.assertTrue((-1, 5) < (0, -5))

    def test_str_operators_across_widths(self) -> None:
        self.assertTrue("a\xff" < "aĀ")
        self.assertTrue("abc" < "abd")
        self.assertFalse("abc" < "ab")
        self.assertTrue("\U0001f600" > "￿")


if __name__ == "__main__":
    unittest.main()
