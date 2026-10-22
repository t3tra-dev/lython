# WHAT: unittest's regular-expression assertions (assertRegex, assertNotRegex,
# assertRaisesRegex) -- passing, and the message each failing one raises --
# and the loader's test name patterns (`-k`), which are fnmatch patterns,
# compared with what CPython's unittest does.
import re
import unittest


class Sample(unittest.TestCase):
    def test_alpha(self) -> None:
        pass

    def test_beta(self) -> None:
        pass

    def test_alphabet(self) -> None:
        pass


class RegexAssertions(unittest.TestCase):
    def test_passing(self) -> None:
        self.assertRegex('hello world', r'w\w+')
        self.assertRegex('hello world', re.compile(r'^h'))
        self.assertNotRegex('hello world', r'\d')
        with self.assertRaisesRegex(ValueError, r'invalid literal'):
            int('x')
        with self.assertRaisesRegex(KeyError, re.compile("'b'")) as cm:
            print({'a': 1}['b'])
        self.assertEqual(str(cm.exception), "'b'")

    def test_regex_failure(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            self.assertRegex('hello', r'\d+')
        self.assertEqual(str(cm.exception), "Regex didn't match: '\\\\d+' not found in 'hello'")

    def test_not_regex_failure(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            self.assertNotRegex('abc123def', r'\d+', 'digits')
        self.assertEqual(str(cm.exception),
                         "Regex matched: '123' matches '\\\\d+' in 'abc123def' : digits")

    def test_raises_regex_mismatch(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            with self.assertRaisesRegex(ValueError, r'^nope'):
                raise ValueError('something else')
        self.assertEqual(str(cm.exception), '"^nope" does not match "something else"')

    def test_raises_regex_not_raised(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            with self.assertRaisesRegex(ValueError, r'x'):
                pass
        self.assertEqual(str(cm.exception), 'ValueError not raised')

    def test_raises_regex_lets_other_exceptions_through(self) -> None:
        with self.assertRaises(TypeError):
            with self.assertRaisesRegex(ValueError, r'x'):
                raise TypeError('x')


class NamePatterns(unittest.TestCase):
    def names(self, patterns: list[str]) -> list[str]:
        loader = unittest.TestLoader()
        loader.testNamePatterns = patterns
        return loader.getTestCaseNames(Sample)

    def test_substring_patterns_are_wrapped_in_stars(self) -> None:
        self.assertEqual(self.names(['*alpha*']), ['test_alpha', 'test_alphabet'])

    def test_wildcards(self) -> None:
        self.assertEqual(self.names(['*.test_alpha']), ['test_alpha'])
        self.assertEqual(self.names(['*test_?eta']), ['test_beta'])
        self.assertEqual(self.names(['*Sample.test_[ab]*']),
                         ['test_alpha', 'test_alphabet', 'test_beta'])
        self.assertEqual(self.names(['*ALPHA*']), [])

    def test_any_pattern_selects(self) -> None:
        self.assertEqual(self.names(['*beta', '*bet']), ['test_alphabet', 'test_beta'])


if __name__ == '__main__':
    unittest.main()
