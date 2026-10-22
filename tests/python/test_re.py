# WHAT: re's matching (anchors, classes, repeats greedy / lazy / possessive,
# groups, backreferences, conditionals, lookaround, atomic groups), its
# flags, search / findall / finditer / sub / subn / split including empty
# matches, Match and Pattern objects, escape, and the errors a bad pattern
# raises, compared with the values CPython 3.14's re returns.
import re
import unittest


def must(m: re.Match | None) -> re.Match:
    if m is None:
        raise AssertionError("no match")
    return m


class Matching(unittest.TestCase):
    def test_match_anchors_at_the_start(self) -> None:
        self.assertIsNone(re.match(r'b', 'ab'))
        self.assertEqual(must(re.match(r'a', 'ab')).span(), (0, 1))

    def test_search_scans(self) -> None:
        m = must(re.search(r'\d+', 'abc 1234 def 56'))
        self.assertEqual(m.group(), '1234')
        self.assertEqual(m.span(), (4, 8))
        self.assertIsNone(re.search(r'\d', 'abc'))

    def test_fullmatch_backtracks_into_alternatives(self) -> None:
        self.assertEqual(must(re.fullmatch(r'a|ab', 'ab')).group(), 'ab')
        self.assertIsNone(re.fullmatch(r'a+', 'aab'))
        self.assertEqual(must(re.fullmatch(r'a*?', 'aaa')).span(), (0, 3))

    def test_groups(self) -> None:
        m = must(re.match(r'(\w+) (\w+)(!)?', 'hello world'))
        self.assertEqual(m.group(0), 'hello world')
        self.assertEqual(m.group(1), 'hello')
        self.assertEqual(m[2], 'world')
        self.assertIsNone(m.group(3))
        self.assertEqual(m.groups(), ('hello', 'world', None))
        self.assertEqual(m.groups('-'), ('hello', 'world', '-'))
        self.assertEqual(m.start(2), 6)
        self.assertEqual(m.end(2), 11)
        self.assertEqual(m.span(3), (-1, -1))
        self.assertEqual(m.lastindex, 2)

    def test_named_groups(self) -> None:
        p = re.compile(r'(?P<first>\w+) (?P<last>\w+)')
        m = must(p.match('Jane Doe'))
        self.assertEqual(m.group('first'), 'Jane')
        self.assertEqual(m['last'], 'Doe')
        self.assertEqual(m.groupdict(), {'first': 'Jane', 'last': 'Doe'})
        self.assertEqual(m.lastgroup, 'last')
        self.assertEqual(p.groupindex, {'first': 1, 'last': 2})
        self.assertEqual(p.groups, 2)

    def test_no_such_group(self) -> None:
        m = must(re.match(r'(a)', 'a'))
        with self.assertRaises(IndexError):
            m.group(2)
        with self.assertRaises(IndexError):
            m.group('nope')

    def test_last_iteration_of_a_group_is_kept(self) -> None:
        m = must(re.match(r'(?:(a)|b)*', 'ab'))
        self.assertEqual(m.group(1), 'a')
        m = must(re.match(r'(a|b)*', 'abb'))
        self.assertEqual(m.group(1), 'b')
        self.assertEqual(m.span(1), (2, 3))

    def test_pos_and_endpos(self) -> None:
        p = re.compile(r'\w+')
        self.assertEqual(must(p.match('ab cd', 3)).group(), 'cd')
        self.assertEqual(must(p.search('abcdef', 1, 3)).group(), 'bc')
        self.assertIsNone(re.compile(r'^b').search('ab', 1))
        m = must(p.match('abc', 1, 2))
        self.assertEqual((m.pos, m.endpos), (1, 2))


class Syntax(unittest.TestCase):
    def test_classes(self) -> None:
        self.assertEqual(re.findall(r'[a-c]', 'abcd'), ['a', 'b', 'c'])
        self.assertEqual(re.findall(r'[^a-c]', 'abcd'), ['d'])
        self.assertEqual(re.findall(r'[\d.]+', 'pi=3.14!'), ['3.14'])
        self.assertEqual(re.findall(r'[]a]', 'a]b'), ['a', ']'])
        self.assertEqual(re.findall(r'[a-]', 'a-b'), ['a', '-'])
        self.assertEqual(re.findall(r'[\s\S]', 'a\n'), ['a', '\n'])
        self.assertEqual(re.findall(r'[^\W\d]+', 'ab1_c'), ['ab', '_c'])

    def test_categories_are_unicode_unless_ascii(self) -> None:
        self.assertEqual(re.findall(r'\w+', 'héllo wörld'), ['héllo', 'wörld'])
        self.assertEqual(re.findall(r'\w+', 'héllo', re.ASCII), ['h', 'llo'])
        self.assertEqual(re.findall(r'\d', '1٣'), ['1', '٣'])
        self.assertEqual(re.findall(r'(?a)\d', '1٣'), ['1'])
        self.assertEqual(re.findall(r'\s', 'a　b'), ['　'])

    def test_dot(self) -> None:
        self.assertEqual(re.findall(r'.', 'a\nb'), ['a', 'b'])
        self.assertEqual(re.findall(r'.', 'a\nb', re.DOTALL), ['a', '\n', 'b'])
        self.assertEqual(re.findall(r'(?s).', 'a\n'), ['a', '\n'])

    def test_anchors(self) -> None:
        self.assertEqual(re.findall(r'^\w', 'ab\ncd'), ['a'])
        self.assertEqual(re.findall(r'^\w', 'ab\ncd', re.M), ['a', 'c'])
        self.assertEqual(re.findall(r'\w$', 'ab\ncd\n'), ['d'])
        self.assertEqual(re.findall(r'\w$', 'ab\ncd', re.MULTILINE), ['b', 'd'])
        self.assertIsNone(re.search(r'a\Z', 'a\n'))
        self.assertIsNotNone(re.search(r'a$', 'a\n'))
        self.assertEqual(re.findall(r'\bfoo\b', 'foo foobar (foo)'), ['foo', 'foo'])
        self.assertEqual(re.findall(r'\Bo', 'foo o'), ['o', 'o'])

    def test_repeats(self) -> None:
        self.assertEqual(must(re.match(r'a{2,3}', 'aaaa')).group(), 'aaa')
        self.assertEqual(must(re.match(r'a{2,3}?', 'aaaa')).group(), 'aa')
        self.assertEqual(must(re.match(r'a{2}', 'aaaa')).group(), 'aa')
        self.assertEqual(must(re.match(r'a{,2}', 'aaaa')).group(), 'aa')
        self.assertEqual(must(re.match(r'a{2,}', 'aaaa')).group(), 'aaaa')
        self.assertEqual(must(re.match(r'x{a', 'x{a')).group(), 'x{a')
        self.assertEqual(must(re.match(r'<.*>', '<a><b>')).group(), '<a><b>')
        self.assertEqual(must(re.match(r'<.*?>', '<a><b>')).group(), '<a>')
        self.assertEqual(must(re.match(r'(ab)+', 'ababc')).group(), 'abab')
        self.assertEqual(must(re.match(r'(ab)+?c', 'ababc')).group(), 'ababc')
        self.assertEqual(must(re.match(r'(?:a|bc)*d', 'abcad')).group(), 'abcad')

    def test_possessive_and_atomic(self) -> None:
        self.assertIsNone(re.match(r'a*+a', 'aaa'))
        self.assertIsNone(re.match(r'(?:ab)++ab', 'ababab'))
        self.assertIsNone(re.match(r'(?>a+)a', 'aaa'))
        self.assertEqual(must(re.match(r'(?>a|ab)c', 'ac')).group(), 'ac')
        self.assertIsNone(re.match(r'(?>a|ab)c', 'abc'))

    def test_backreferences(self) -> None:
        self.assertEqual(must(re.search(r'(\w)\1', 'abccd')).group(), 'cc')
        self.assertEqual(must(re.search(r'(?P<q>[\'"]).*?(?P=q)', 'say "hi" now')).group(), '"hi"')
        self.assertEqual(must(re.match(r'(?i)(a)\1', 'aA')).group(), 'aA')

    def test_conditional(self) -> None:
        p = re.compile(r'(<)?\w+(?(1)>|$)')
        self.assertEqual(must(p.match('<user>')).group(), '<user>')
        self.assertEqual(must(p.match('user')).group(), 'user')
        self.assertIsNone(p.match('<user'))

    def test_lookaround(self) -> None:
        self.assertEqual(re.findall(r'\w+(?=,)', 'a, b, c'), ['a', 'b'])
        self.assertEqual(re.findall(r'\b\w+\b(?!,)', 'a, b, c'), ['c'])
        self.assertEqual(re.findall(r'(?<=\$)\d+', 'cost $12 or $3'), ['12', '3'])
        self.assertEqual(re.findall(r'(?<!\$)\b\d+', '$12 34'), ['34'])

    def test_verbose(self) -> None:
        p = re.compile(r'''
            (\d+)   # whole part
            \.      # point
            (\d*)   # fraction
        ''', re.VERBOSE)
        self.assertEqual(must(p.match('3.14')).groups(), ('3', '14'))

    def test_inline_and_scoped_flags(self) -> None:
        self.assertIsNotNone(re.match(r'(?i)abc', 'ABC'))
        self.assertIsNotNone(re.match(r'a(?i:b)c', 'aBc'))
        self.assertIsNone(re.match(r'a(?i:b)c', 'aBC'))
        self.assertIsNone(re.match(r'(?i)a(?-i:b)', 'AB'))

    def test_comments_and_escapes(self) -> None:
        self.assertEqual(must(re.match(r'a(?#note)b', 'ab')).group(), 'ab')
        self.assertEqual(must(re.match(r'\x41B\101\t', 'ABA\t')).group(), 'ABA\t')
        self.assertEqual(must(re.match(r'\.\*', '.*')).group(), '.*')


class IgnoreCase(unittest.TestCase):
    def test_ascii_letters(self) -> None:
        self.assertEqual(re.findall(r'[a-z]+', 'Hello WORLD', re.I), ['Hello', 'WORLD'])
        self.assertIsNotNone(re.match(r'K', 'k', re.IGNORECASE))

    def test_unicode_case_folding(self) -> None:
        self.assertIsNotNone(re.match(r's', 'ſ', re.I))
        self.assertIsNotNone(re.match(r'k', 'K', re.I))
        self.assertIsNotNone(re.match(r'[a-z]', 'K', re.I))
        self.assertIsNone(re.match(r'k', 'K', re.I | re.A))
        self.assertIsNotNone(re.match(r'ΣΑΣ', 'σας', re.I))


class Iteration(unittest.TestCase):
    def test_findall_with_one_group(self) -> None:
        self.assertEqual(re.findall(r'(\w)=\d', 'a=1 b=2'), ['a', 'b'])
        self.assertEqual(re.findall(r'(x)?y', 'y xy'), ['', 'x'])

    def test_empty_matches(self) -> None:
        self.assertEqual(re.findall(r'a*', 'baaa'), ['', 'aaa', ''])
        self.assertEqual(re.findall(r'\b', 'ab cd'), ['', '', '', ''])
        spans = [m.span() for m in re.finditer(r'a*', 'baaab')]
        self.assertEqual(spans, [(0, 0), (1, 4), (4, 4), (5, 5)])

    def test_finditer(self) -> None:
        found = [(m.group(), m.start()) for m in re.finditer(r'\d+', 'a1b22c333')]
        self.assertEqual(found, [('1', 1), ('22', 3), ('333', 6)])


class Substitution(unittest.TestCase):
    def test_sub_with_templates(self) -> None:
        self.assertEqual(re.sub(r'(\w+)@(\w+)', r'\2 at \1', 'me@host you@there'),
                         'host at me there at you')
        self.assertEqual(re.sub(r'(?P<w>\w+)', r'<\g<w>>', 'a b'), '<a> <b>')
        self.assertEqual(re.sub(r'\w+', r'[\g<0>]', 'a b'), '[a] [b]')
        self.assertEqual(re.sub(r'x', r'\n\\', 'axb'), 'a\n\\b')
        self.assertEqual(re.sub(r'(a)|b', r'[\1]', 'ab'), '[a][]')

    def test_sub_with_a_function(self) -> None:
        def upper(m: re.Match) -> str:
            text = m.group()
            return text.upper() if text is not None else ''
        self.assertEqual(re.sub(r'[aeiou]', upper, 'banana'), 'bAnAnA')

    def test_sub_count_and_subn(self) -> None:
        self.assertEqual(re.sub(r'a', 'x', 'aaaa', count=2), 'xxaa')
        self.assertEqual(re.subn(r'a', 'x', 'banana'), ('bxnxnx', 3))
        self.assertEqual(re.compile(r'\s+').sub(' ', 'a  b \t c'), 'a b c')

    def test_sub_empty_matches(self) -> None:
        self.assertEqual(re.sub(r'x*', '-', 'abxd'), '-a-b--d-')
        self.assertEqual(re.sub(r'', '-', 'abc'), '-a-b-c-')

    def test_expand(self) -> None:
        m = must(re.match(r'(\w+) (?P<x>\w+)', 'hello world'))
        self.assertEqual(m.expand(r'\2 \1 \g<x>'), 'world hello world')

    def test_bad_templates(self) -> None:
        with self.assertRaises(re.error):
            re.sub(r'(a)', r'\2', 'a')
        with self.assertRaises(IndexError):
            re.sub(r'(a)', r'\g<nope>', 'a')


class Splitting(unittest.TestCase):
    def test_split(self) -> None:
        self.assertEqual(re.split(r'\s*,\s*', 'a , b,c ,d'), ['a', 'b', 'c', 'd'])
        self.assertEqual(re.split(r'(,)', 'a,b'), ['a', ',', 'b'])
        self.assertEqual(re.split(r'(,)|(;)', 'a,b;c'), ['a', ',', None, 'b', None, ';', 'c'])
        self.assertEqual(re.split(r',', 'a,b,c,d', maxsplit=2), ['a', 'b', 'c,d'])
        self.assertEqual(re.split(r'x*', 'axbc'), ['', 'a', '', 'b', 'c', ''])
        self.assertEqual(re.split(r'\W+', 'Words, words, words.'), ['Words', 'words', 'words', ''])


class Objects(unittest.TestCase):
    def test_match_repr(self) -> None:
        self.assertEqual(repr(re.match(r'a+', 'aab')),
                         "<re.Match object; span=(0, 2), match='aa'>")
        self.assertEqual(repr(re.match(r'.*', 'x' * 60)),
                         "<re.Match object; span=(0, 60), match='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx>")

    def test_pattern_repr(self) -> None:
        self.assertEqual(repr(re.compile(r'a+')), "re.compile('a+')")
        self.assertEqual(repr(re.compile(r'a', re.I | re.M)),
                         "re.compile('a', re.IGNORECASE|re.MULTILINE)")
        self.assertEqual(repr(re.compile(r'a', re.A)), "re.compile('a', re.ASCII)")
        self.assertEqual(re.compile(r'a', re.I).flags, re.I | re.UNICODE)

    def test_pattern_equality(self) -> None:
        self.assertEqual(re.compile(r'a+'), re.compile(r'a+'))
        self.assertNotEqual(re.compile(r'a+'), re.compile(r'a+', re.I))
        self.assertEqual(re.compile(r'a+').pattern, 'a+')

    def test_compiled_pattern_passes_through(self) -> None:
        p = re.compile(r'\d')
        self.assertIs(re.compile(p), p)
        self.assertEqual(re.findall(p, 'a1b2'), ['1', '2'])
        with self.assertRaises(ValueError):
            re.compile(p, re.I)

    def test_escape(self) -> None:
        self.assertEqual(re.escape('a.b*c'), 'a\\.b\\*c')
        self.assertEqual(re.escape('1+1=2 (ok)'), '1\\+1=2\\ \\(ok\\)')
        self.assertEqual(re.escape('_x-y'), '_x\\-y')
        self.assertIsNotNone(re.fullmatch(re.escape('[a]{1}'), '[a]{1}'))


class Errors(unittest.TestCase):
    def check(self, pattern: str, message: str, pos: int | None) -> None:
        with self.assertRaises(re.error) as cm:
            re.compile(pattern)
        self.assertEqual(str(cm.exception), message)
        self.assertEqual(cm.exception.pos, pos)
        self.assertEqual(cm.exception.pattern, pattern if pos is not None else None)

    def test_messages(self) -> None:
        cases: list[tuple[str, str, int | None]] = [
            ('(a', 'missing ), unterminated subpattern at position 0', 0),
            ('a)', 'unbalanced parenthesis at position 1', 1),
            ('*a', 'nothing to repeat at position 0', 0),
            ('a**', 'multiple repeat at position 2', 2),
            ('[a', 'unterminated character set at position 0', 0),
            ('[z-a]', 'bad character range z-a at position 1', 1),
            (r'\q', 'bad escape \\q at position 0', 0),
            (r'(?P<1a>x)', "bad character in group name '1a' at position 4", 4),
            (r'(?P=n)', "unknown group name 'n' at position 4", 4),
            (r'(a)\2', 'invalid group reference 2 at position 4', 4),
            (r'(?<=a+)b', 'look-behind requires fixed-width pattern', None),
            ('a{3,1}', 'min repeat greater than max repeat at position 2', 2),
            ('(?z)', 'unknown extension ?z at position 1', 1),
        ]
        for pattern, message, pos in cases:
            self.check(pattern, message, pos)

    def test_line_and_column(self) -> None:
        with self.assertRaises(re.error) as cm:
            re.compile('a\n(b')
        self.assertEqual(str(cm.exception),
                         'missing ), unterminated subpattern at position 2 (line 2, column 1)')
        self.assertEqual(cm.exception.lineno, 2)
        self.assertEqual(cm.exception.colno, 1)


if __name__ == '__main__':
    unittest.main()
