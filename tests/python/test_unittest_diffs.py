# WHAT: the differences a failing assertEqual shows for a list, a tuple, a
# dict, a set and a multi-line str -- difflib.ndiff over pprint.pformat --
# and maxDiff cutting a long one short, compared with the text CPython's
# unittest produces.
import unittest


class Unlimited(unittest.TestCase):
    maxDiff = None


class Limited(unittest.TestCase):
    maxDiff = 40


class DiffMessages(unittest.TestCase):
    def test_list(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            Unlimited().assertEqual([1, 2, 3, 4], [1, 2, 5, 4])
        self.assertEqual(
            str(cm.exception),
            'Lists differ: [1, 2, 3, 4] != [1, 2, 5, 4]\n\nFirst differing '
            'element 2:\n3\n5\n\n- [1, 2, 3, 4]\n?        ^\n\n'
            '+ [1, 2, 5, 4]\n?        ^\n')

    def test_list_longer(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            Unlimited().assertEqual([1, 2], [1, 2, 3])
        self.assertEqual(
            str(cm.exception),
            'Lists differ: [1, 2] != [1, 2, 3]\n\nSecond list contains 1 '
            'additional elements.\nFirst extra element 2:\n3\n\n- [1, 2]\n'
            '+ [1, 2, 3]\n?      +++\n')

    def test_tuple(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            Unlimited().assertEqual((1, "a"), (1, "b"))
        self.assertEqual(
            str(cm.exception),
            "Tuples differ: (1, 'a') != (1, 'b')\n\nFirst differing element "
            "1:\n'a'\n'b'\n\n- (1, 'a')\n?      ^\n\n+ (1, 'b')\n?      ^\n")

    def test_dict(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            Unlimited().assertEqual({"a": 1, "b": [1, 2]}, {"a": 1, "b": [1, 3]})
        self.assertEqual(
            str(cm.exception),
            "{'a': 1, 'b': [1, 2]} != {'a': 1, 'b': [1, 3]}\n"
            "- {'a': 1, 'b': [1, 2]}\n?                   ^\n\n"
            "+ {'a': 1, 'b': [1, 3]}\n?                   ^\n")

    def test_set(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            Unlimited().assertEqual({1, 2, 3}, {2, 3, 4})
        self.assertEqual(
            str(cm.exception),
            'Items in the first set but not the second:\n1\n'
            'Items in the second set but not the first:\n4')

    def test_multiline(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            Unlimited().assertEqual("one\ntwo\nthree\n", "one\ntoo\nthree\n")
        self.assertEqual(
            str(cm.exception),
            "'one\\ntwo\\nthree\\n' != 'one\\ntoo\\nthree\\n'\n  one\n- two\n"
            "?  ^\n+ too\n?  ^\n  three\n")

    def test_one_line_str(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            Unlimited().assertEqual("abc", "abd")
        self.assertEqual(str(cm.exception),
                         "'abc' != 'abd'\n- abc\n?   ^\n+ abd\n?   ^\n")

    def test_long_list_is_laid_out_one_element_per_line(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            Unlimited().assertEqual(list(range(30)), list(range(1, 31)))
        lines = str(cm.exception).split("\n")
        self.assertEqual(lines[:8], [
            'Lists differ: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,[63 '
            'chars], 29] != [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13[64 '
            'chars], 30]',
            '', 'First differing element 0:', '0', '1', '', '+ [1,', '- [0,'])
        self.assertEqual(lines[-8:], ['   28,', '-  29]', '?    ^', '',
                                      '+  29,', '?    ^', '', '+  30]'])

    def test_max_diff_cuts_the_diff(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            Limited().assertEqual(list(range(30)), list(range(1, 31)))
        self.assertEqual(
            str(cm.exception).split("\n")[-1],
            'Diff is 236 characters long. Set self.maxDiff to None to see it.')


if __name__ == "__main__":
    unittest.main()
