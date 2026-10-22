# WHAT: difflib's SequenceMatcher (matching blocks, opcodes, ratios, junk and
# the popular-element heuristic), Differ through ndiff, unified_diff,
# context_diff, restore and get_close_matches, compared with the values
# CPython 3.14's difflib returns for the same inputs.
import difflib
import unittest


def isblank(x: str) -> bool:
    return x == " "


TEXT1 = '''  1. Beautiful is better than ugly.
  2. Explicit is better than implicit.
  3. Simple is better than complex.
  4. Complex is better than complicated.
'''
TEXT2 = '''  1. Beautiful is better than ugly.
  3.   Simple is better than complex.
  4. Complicated is better than complex.
  5. Flat is better than nested.
'''


class Difflib(unittest.TestCase):
    def test_ndiff_zen(self) -> None:
        self.assertEqual(
            list(difflib.ndiff(TEXT1.splitlines(keepends=True), TEXT2.splitlines(keepends=True))),
            ['    1. Beautiful is better than ugly.\n', '-   2. Explicit is better than implicit.\n', '-   3. Simple is better than complex.\n', '+   3.   Simple is better than complex.\n', '?     ++\n', '-   4. Complex is better than complicated.\n', '+   4. Complicated is better than complex.\n', '+   5. Flat is better than nested.\n'])

    def test_ndiff_short(self) -> None:
        self.assertEqual(
            list(difflib.ndiff(['one\n', 'two\n', 'three\n'], ['ore\n', 'tree\n', 'emu\n'])),
            ['- one\n', '?  ^\n', '+ ore\n', '?  ^\n', '- two\n', '- three\n', '?  -\n', '+ tree\n', '+ emu\n'])

    def test_ndiff_tabs(self) -> None:
        self.assertEqual(
            list(difflib.ndiff(['\tabcDefghiJkl\n'], ['\tabcdefGhijkl\n'])),
            ['- \tabcDefghiJkl\n', '? \t   ^  ^  ^\n', '+ \tabcdefGhijkl\n', '? \t   ^  ^  ^\n'])

    def test_matching_blocks(self) -> None:
        self.assertEqual(
            [(m.a, m.b, m.size) for m in difflib.SequenceMatcher(None, 'abxcd', 'abcd').get_matching_blocks()],
            [(0, 0, 2), (3, 2, 2), (5, 4, 0)])

    def test_ratios(self) -> None:
        self.assertEqual(
            (difflib.SequenceMatcher(None, 'abcd', 'bcde').ratio(), difflib.SequenceMatcher(None, 'abcd', 'bcde').quick_ratio(), difflib.SequenceMatcher(None, 'abcd', 'bcde').real_quick_ratio()),
            (0.75, 0.75, 1.0))

    def test_opcodes(self) -> None:
        self.assertEqual(
            difflib.SequenceMatcher(None, 'qabxcd', 'abycdf').get_opcodes(),
            [('delete', 0, 1, 0, 0), ('equal', 1, 3, 0, 2), ('replace', 3, 4, 2, 3), ('equal', 4, 6, 3, 5), ('insert', 6, 6, 5, 6)])

    def test_junk_longest(self) -> None:
        self.assertEqual(
            str(difflib.SequenceMatcher(isblank, ' abcd', 'abcd abcd').find_longest_match(0, 5, 0, 9)),
            'Match(a=1, b=0, size=4)')

    def test_longest(self) -> None:
        self.assertEqual(
            str(difflib.SequenceMatcher(None, ' abcd', 'abcd abcd').find_longest_match(0, 5, 0, 9)),
            'Match(a=0, b=4, size=5)')

    def test_unified(self) -> None:
        self.assertEqual(
            list(difflib.unified_diff('one two three four'.split(), 'zero one tree four'.split(), 'Original', 'Current', '2005-01-26 23:30:50', '2010-04-02 10:20:52', lineterm='')),
            ['--- Original\t2005-01-26 23:30:50', '+++ Current\t2010-04-02 10:20:52', '@@ -1,4 +1,4 @@', '+zero', ' one', '-two', '-three', '+tree', ' four'])

    def test_context(self) -> None:
        self.assertEqual(
            list(difflib.context_diff('one\ntwo\nthree\nfour\n'.splitlines(True), 'zero\none\ntree\nfour\n'.splitlines(True), 'Original', 'Current')),
            ['*** Original\n', '--- Current\n', '***************\n', '*** 1,4 ****\n', '  one\n', '! two\n', '! three\n', '  four\n', '--- 1,4 ----\n', '+ zero\n', '  one\n', '! tree\n', '  four\n'])

    def test_close_matches(self) -> None:
        self.assertEqual(
            difflib.get_close_matches('appel', ['ape', 'apple', 'peach', 'puppy']),
            ['apple', 'ape'])

    def test_restore1(self) -> None:
        self.assertEqual(
            list(difflib.restore(list(difflib.ndiff(['one\n', 'two\n'], ['ore\n', 'tree\n'])), 1)),
            ['one\n', 'two\n'])

    def test_restore2(self) -> None:
        self.assertEqual(
            list(difflib.restore(list(difflib.ndiff(['one\n', 'two\n'], ['ore\n', 'tree\n'])), 2)),
            ['ore\n', 'tree\n'])

    def test_junk(self) -> None:
        self.assertEqual(
            (difflib.IS_LINE_JUNK('  #   \n'), difflib.IS_LINE_JUNK('hello\n'), difflib.IS_CHARACTER_JUNK('\t'), difflib.IS_CHARACTER_JUNK('x')),
            (True, False, True, False))

    def test_autojunk(self) -> None:
        self.assertEqual(
            difflib.SequenceMatcher(None, ['x'] * 250 + ['a', 'b'], ['x'] * 240 + ['a', 'c']).get_opcodes(),
            [('delete', 0, 10, 0, 0), ('equal', 10, 251, 0, 241), ('replace', 251, 252, 241, 242)])

    def test_no_autojunk(self) -> None:
        self.assertEqual(
            difflib.SequenceMatcher(None, ['x'] * 250 + ['a', 'b'], ['x'] * 240 + ['a', 'c'], autojunk=False).get_opcodes(),
            [('delete', 0, 10, 0, 0), ('equal', 10, 251, 0, 241), ('replace', 251, 252, 241, 242)])


if __name__ == "__main__":
    unittest.main()
