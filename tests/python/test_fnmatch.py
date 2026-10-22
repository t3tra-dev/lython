# WHAT: fnmatch's shell patterns (*, ?, sets, negated sets, ranges, empty
# and reversed ranges, set operators, a lone '['), the regular expressions
# translate() writes for them, and filter / filterfalse, compared with the
# values CPython 3.14's fnmatch returns.
import fnmatch
import unittest


class FnmatchCase(unittest.TestCase):
    def check(self, name: str, pattern: str, expected: bool = True) -> None:
        self.assertEqual(fnmatch.fnmatchcase(name, pattern), expected,
                         '%r %r' % (name, pattern))

    def test_wildcards(self) -> None:
        self.check('abc', 'abc')
        self.check('abc', '?*?')
        self.check('abc', '???*')
        self.check('abc', '*???')
        self.check('abc', '???')
        self.check('abc', '*')
        self.check('abc', 'ab[cd]')
        self.check('abc', 'ab[!de]')
        self.check('abc', 'ab[de]', False)
        self.check('a', '??', False)
        self.check('a', 'b', False)
        self.check('\nfoo', 'foo', False)
        self.check('foo\n', 'foo', False)
        self.check('\n', '\n')
        self.check('foo\nbar', 'foo*')
        self.check('foo\nbar\n', 'foo*')
        self.check('\nfoo', '*foo')

    def test_case_is_kept(self) -> None:
        self.check('abc', 'ABC', False)
        self.check('AbC', 'AbC')
        self.assertTrue(fnmatch.fnmatch('abc', 'a?c'))

    def test_sets(self) -> None:
        self.check('[', '[[]')
        self.check('&', '[a&&b]')
        self.check('|', '[a||b]')
        self.check('~', '[a~~b]')
        self.check(',', '[a-z+--A-Z]')
        self.check('.', '[a-z--/A-Z]')
        self.check('b', '[a-c]')
        self.check('-', '[a-]')
        self.check('d', '[!a-c]')
        self.check('b', '[!a-c]', False)
        self.check('a', '[z-a]', False)
        self.check('a', '[]', False)
        self.check('[]', '[]')
        self.check('[!]', '[!]')
        self.check('a', '[!]', False)
        self.check('\\', r'[\]')
        self.check('a', r'[!\]')
        self.check('\\', r'[!\]', False)

    def test_star_runs(self) -> None:
        self.check('a' * 50, '*a*a*a*a*a*a*a*a*a*a')
        self.check('a' * 50 + 'b', '*a*a*a*a*a*a*a*a*a*a', False)
        self.check('abcdef', '*b*d*f')
        self.check('abcdef', '*b*d*e', False)


class Translate(unittest.TestCase):
    def test_translate(self) -> None:
        self.assertEqual(fnmatch.translate('*'), r'(?s:.*)\z')
        self.assertEqual(fnmatch.translate('?'), r'(?s:.)\z')
        self.assertEqual(fnmatch.translate('a?b*'), r'(?s:a.b.*)\z')
        self.assertEqual(fnmatch.translate('[abc]'), r'(?s:[abc])\z')
        self.assertEqual(fnmatch.translate('[]]'), r'(?s:[]])\z')
        self.assertEqual(fnmatch.translate('[!x]'), r'(?s:[^x])\z')
        self.assertEqual(fnmatch.translate('[^x]'), r'(?s:[\^x])\z')
        self.assertEqual(fnmatch.translate('[x'), r'(?s:\[x)\z')
        self.assertEqual(fnmatch.translate('**a*b*'), r'(?s:(?>.*?a)(?>.*?b).*)\z')
        self.assertEqual(fnmatch.translate('*.py'), r'(?s:.*\.py)\z')
        self.assertEqual(fnmatch.translate('[a-z--/A-Z]'), r'(?s:[a-z\--/A-Z])\z')
        self.assertEqual(fnmatch.translate('[a&&b]'), r'(?s:[a\&\&b])\z')


class Filter(unittest.TestCase):
    def test_filter(self) -> None:
        names = ['a.py', 'b.txt', 'c.py', 'Dpy']
        self.assertEqual(fnmatch.filter(names, '*.py'), ['a.py', 'c.py'])
        self.assertEqual(fnmatch.filterfalse(names, '*.py'), ['b.txt', 'Dpy'])
        self.assertEqual(fnmatch.filter([], '*'), [])


if __name__ == '__main__':
    unittest.main()
