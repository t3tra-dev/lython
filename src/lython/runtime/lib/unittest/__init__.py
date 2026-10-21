"""
Python unit testing framework, based on Erich Gamma's JUnit and Kent Beck's
Smalltalk testing framework (used with permission).

This module contains the core framework classes that form the basis of
specific test cases and suites (TestCase, TestSuite etc.), and also a
text-based utility class for running the tests and reporting the results
 (TextTestRunner).

Simple usage:

    import unittest

    class IntegerArithmeticTestCase(unittest.TestCase):
        def testAdd(self):  # test method names begin with 'test'
            self.assertEqual((1 + 2), 3)
            self.assertEqual(0 + 1, 1)
        def testMultiply(self):
            self.assertEqual((0 * 10), 0)
            self.assertEqual((5 * 8), 40)

    if __name__ == '__main__':
        unittest.main()

This is Lython's port of CPython 3.14's Lib/unittest, with the same modules
(case, result, suite, loader, runner, main, util) and the same names in them,
restricted to the well-typed statically compilable surface.

Deviations from CPython:
- a skipped / expected-failure test is a WRAPPED method, not a function with
  `__unittest_skip__` attributes on it; class-level @skip is not provided.
  Its docstring is the wrapper's (None), where CPython's skip copies the
  test's through functools.wraps and expectedFailure returns the test itself.
- assertEqual dispatches on the static type of its arguments (a generic
  method), not on a `_type_equality_funcs` dict keyed by run-time class.
- assertRaises' callable form is assertRaisesCall (one name cannot be both a
  context manager factory and a call forwarder here).
- cleanups are stored as closures over their arguments, not as triples.
- a subtest's params are a dict holding its own keys first and then its
  parent's others, not an _OrderedChainMap over the parent's.
- module fixtures (setUpModule) are not provided, and loadTestsFromModule
  takes the module's NAME: it finds the TestCase subclasses through
  `TestCase.__subclasses__()` whose `__module__` it is, in the order of their
  names -- `dir(module)`'s order -- rather than reading the module's
  namespace, so a TestCase imported into the module is not one of its tests.
- `-k` matches a substring, not an fnmatch pattern; command-line test names,
  discovery, `-c` (signals) and `-b` (buffering) are not provided.
- failure messages carry no difflib/pprint diff, and output is never colored.
- a stream is an `io.TextIOWrapper | io.StringIO`, not any object with write.
- a FunctionTestCase compares and hashes as any TestCase does (by class and
  method name, not by its functions), and its repr names the function rather
  than showing it: a function value here has no identity to compare or
  address to print.
- frames of this package are dropped from a traceback by FILE NAME, not by a
  `__unittest` global in the frame.
"""

__all__ = ['TestResult', 'TestCase', 'TestSuite',
           'TextTestRunner', 'TestLoader', 'FunctionTestCase', 'main',
           'defaultTestLoader', 'SkipTest', 'skip', 'skipIf', 'skipUnless',
           'expectedFailure', 'TextTestResult',
           'addModuleCleanup', 'doModuleCleanups']

__unittest = True

from .result import TestResult
from .case import (addModuleCleanup, TestCase, FunctionTestCase, SkipTest, skip,
                   skipIf, skipUnless, expectedFailure, doModuleCleanups)
from .suite import BaseTestSuite, TestSuite  # noqa: F401
from .loader import TestLoader, defaultTestLoader
from .main import TestProgram, main  # noqa: F401
from .runner import TextTestRunner, TextTestResult
