"""A well-typed port of CPython 3.14's Lib/unittest (case, result, suite,
loader, runner, util and a minimal main), as one module.

A measuring fixture for docs/unittest-port.md, not the module that will ship.
It runs under CPython too -- next to a test file it shadows the standard
unittest -- so the same test file can be run both ways and compared:

    python3.14 t01_pass.py          lyc jit t01_pass.py

Every place that is written the way it is only because Lython refuses the
plain spelling is marked `# WORKAROUND(<probe>)`, naming the probe in
tests/probe that records the refusal. When a probe agrees with CPython, its
workaround can go.

Not here yet, each for a recorded reason:
- subTest: the generator does not compile
  (ut_bug_class_body_field_object_into_inner_contextmanager), so _SubTest and
  the subTest arms of testPartExecutor are unused.
- cleanups are DISCARDED, not run, and there is no doCleanups
  (ut_bug_exc_info_after_calling_a_callable_from_a_list).
- TestCase.debug, which the runner does not use.

Deviations from CPython that the static types themselves call for:
- a skipped / expected-failure test is a WRAPPED method, not a function with
  `__unittest_skip__` attributes on it; class-level @skip is not provided.
- assertEqual dispatches on the static type of its arguments (a generic
  method), not on a `_type_equality_funcs` dict keyed by run-time class.
- assertRaises' callable form is assertRaisesCall (one name cannot be both a
  context manager factory and a call forwarder here).
- cleanups are stored as closures over their arguments, not as triples.
- module fixtures (setUpModule) and the module loader are not provided; the
  tests to run are named explicitly: main(TestA, TestB).
- failure messages carry no difflib/pprint diff.
- frames of this module are dropped from a traceback by FILE NAME, not by a
  `__unittest` global in the frame.
"""

import contextlib
import sys
import time
import traceback
import types
from collections.abc import Callable, Iterator

_SysExcInfoType = tuple[type[BaseException], BaseException, types.TracebackType]

_MAX_LENGTH: int = 80
_PLACEHOLDER_LEN: int = 12
_MIN_BEGIN_LEN: int = 5
_MIN_END_LEN: int = 5
_MIN_COMMON_LEN: int = 5
# WORKAROUND(ut_unsupported_computed_constant_of_imported_module): a computed constant of an imported module needs an annotation.
_MIN_DIFF_LEN: int = _MAX_LENGTH - (_MIN_BEGIN_LEN + _PLACEHOLDER_LEN + _MIN_COMMON_LEN
                                    + _PLACEHOLDER_LEN + _MIN_END_LEN)


# ---------------------------------------------------------------- util

def _shorten(s: str, prefixlen: int, suffixlen: int) -> str:
    skip = len(s) - prefixlen - suffixlen
    if skip > _PLACEHOLDER_LEN:
        s = '%s[%d chars]%s' % (s[:prefixlen], skip, s[len(s) - suffixlen:])
    return s


def _commonprefix(m: list[str]) -> str:
    if not m:
        return ''
    s1 = min(m)
    s2 = max(m)
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1


def _common_shorten_repr(first: str, second: str) -> tuple[str, str]:
    maxlen = max(len(first), len(second))
    if maxlen <= _MAX_LENGTH:
        return first, second
    prefix = _commonprefix([first, second])
    prefixlen = len(prefix)
    common_len = _MAX_LENGTH - (maxlen - prefixlen + _MIN_BEGIN_LEN + _PLACEHOLDER_LEN)
    if common_len > _MIN_COMMON_LEN:
        prefix = _shorten(prefix, _MIN_BEGIN_LEN, common_len)
        return prefix + first[prefixlen:], prefix + second[prefixlen:]
    prefix = _shorten(prefix, _MIN_BEGIN_LEN, _MIN_COMMON_LEN)
    return (prefix + _shorten(first[prefixlen:], _MIN_DIFF_LEN, _MIN_END_LEN),
            prefix + _shorten(second[prefixlen:], _MIN_DIFF_LEN, _MIN_END_LEN))


def safe_repr(obj: object, short: bool = False) -> str:
    result = repr(obj)
    if not short or len(result) < _MAX_LENGTH:
        return result
    return result[:_MAX_LENGTH] + ' [truncated]...'


def strclass(cls: type) -> str:
    return "%s.%s" % (cls.__module__, cls.__qualname__)


# ---------------------------------------------------------------- case

class SkipTest(Exception):
    """Raise this exception in a test to skip it."""


class _ShouldStop(Exception):
    """The test should stop."""


class _UnexpectedSuccess(Exception):
    """The test was supposed to fail, but it didn't!"""


class _ExpectedFailure(Exception):
    """The test failed as its decorator said it would."""

    def __init__(self, exc_info: _SysExcInfoType) -> None:
        super().__init__()
        self.exc_info = exc_info


class _Outcome:
    def __init__(self, result: TestResult) -> None:
        self.expecting_failure = False
        self.result = result
        self.success = True
        self.expectedFailure: _SysExcInfoType | None = None
        self.unexpectedSuccess = False

    @contextlib.contextmanager
    def testPartExecutor(self, test_case: TestCase, subTest: bool = False) -> Iterator[None]:
        old_success = self.success
        self.success = True
        try:
            yield
        except KeyboardInterrupt:
            raise
        except SkipTest as e:
            self.success = False
            self.result.addSkip(test_case, str(e))
        except _ShouldStop:
            pass
        except _ExpectedFailure as e:
            self.expectedFailure = e.exc_info
        except _UnexpectedSuccess:
            self.unexpectedSuccess = True
        except:
            exc_info = sys.exc_info()
            if self.expecting_failure:
                self.expectedFailure = exc_info
            else:
                self.success = False
                if subTest and isinstance(test_case, _SubTest):
                    self.result.addSubTest(test_case.test_case, test_case, exc_info)
                elif issubclass(exc_info[0], test_case.failureException):
                    self.result.addFailure(test_case, exc_info)
                else:
                    self.result.addError(test_case, exc_info)
        else:
            if subTest and self.success and isinstance(test_case, _SubTest):
                self.result.addSubTest(test_case.test_case, test_case, None)
        finally:
            self.success = self.success and old_success


def skip[T: TestCase](reason: str) -> Callable[[Callable[[T], None]], Callable[[T], None]]:
    """Unconditionally skip a test."""
    def decorator(test_item: Callable[[T], None]) -> Callable[[T], None]:
        def skip_wrapper(self: T) -> None:
            raise SkipTest(reason)
        return skip_wrapper
    return decorator


def skipIf[T: TestCase](condition: bool, reason: str) -> Callable[[Callable[[T], None]], Callable[[T], None]]:
    """Skip a test if the condition is true."""
    def decorator(test_item: Callable[[T], None]) -> Callable[[T], None]:
        if condition:
            def skip_wrapper(self: T) -> None:
                raise SkipTest(reason)
            return skip_wrapper
        return test_item
    return decorator


def skipUnless[T: TestCase](condition: bool, reason: str) -> Callable[[Callable[[T], None]], Callable[[T], None]]:
    """Skip a test unless the condition is true."""
    # WORKAROUND(ut_unsupported_generic_factory_result_returned): returning skipIf(not condition, reason) is refused
    # (a generic factory's result handed on unapplied).
    def decorator(test_item: Callable[[T], None]) -> Callable[[T], None]:
        if not condition:
            def skip_wrapper(self: T) -> None:
                raise SkipTest(reason)
            return skip_wrapper
        return test_item
    return decorator


def expectedFailure[T: TestCase](test_item: Callable[[T], None]) -> Callable[[T], None]:
    def wrapper(self: T) -> None:
        try:
            test_item(self)
        except SkipTest:
            raise
        except Exception:
            raise _ExpectedFailure(sys.exc_info())
        raise _UnexpectedSuccess()
    return wrapper


class _AssertRaisesContext[E: BaseException]:
    """A context manager used to implement TestCase.assertRaises* methods."""

    def __init__(self, expected: type[E], test_case: TestCase, msg: str | None = None) -> None:
        self.expected = expected
        self.test_case = test_case
        self.msg = msg
        self.obj_name: str | None = None
        self._exception: E | None = None

    @property
    def exception(self) -> E:
        exc = self._exception
        if exc is None:
            raise AttributeError("'_AssertRaisesContext' object has no attribute 'exception'")
        return exc

    def __enter__(self) -> _AssertRaisesContext[E]:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None,
                 tb: types.TracebackType | None) -> bool:
        if exc_type is None:
            exc_name = self.expected.__name__
            obj_name = self.obj_name
            if obj_name is not None:
                self.test_case.fail(self.test_case._formatMessage(
                    self.msg, "{} not raised by {}".format(exc_name, obj_name)))
            else:
                self.test_case.fail(self.test_case._formatMessage(
                    self.msg, "{} not raised".format(exc_name)))
            return False
        if not issubclass(exc_type, self.expected):
            # let unexpected exceptions pass through
            return False
        if isinstance(exc_value, self.expected):
            self._exception = exc_value
        return True


class TestCase:
    """A class whose instances are single test cases."""

    failureException: type[BaseException] = AssertionError
    longMessage: bool = True
    maxDiff: int | None = 80 * 8
    _diffThreshold: int = 2 ** 16

    def __init__(self, methodName: str = 'runTest') -> None:
        self._testMethodName = methodName
        self._outcome: _Outcome | None = None
        if methodName != 'runTest' and not hasattr(self, methodName):
            raise ValueError("no such test method in %s: %s" % (type(self), methodName))
        self._cleanups: list[Callable[[], object]] = []
        self._subtest: _SubTest | None = None

    def addCleanup[**P](self, function: Callable[P, object], /, *args: P.args, **kwargs: P.kwargs) -> None:
        """Add a function, with arguments, to be called when the test is
        completed. Functions added are called on a LIFO basis and are
        called after tearDown on test failure or success."""
        self._cleanups.append(lambda: function(*args, **kwargs))

    def setUp(self) -> None:
        "Hook method for setting up the test fixture before exercising it."
        pass

    def tearDown(self) -> None:
        "Hook method for deconstructing the test fixture after testing it."
        pass

    @classmethod
    def setUpClass(cls) -> None:
        "Hook method for setting up class fixture before running tests in the class."

    @classmethod
    def tearDownClass(cls) -> None:
        "Hook method for deconstructing the class fixture after running all tests in the class."

    def countTestCases(self) -> int:
        return 1

    def shortDescription(self) -> str | None:
        return None

    def id(self) -> str:
        return "%s.%s" % (strclass(type(self)), self._testMethodName)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TestCase):
            return False
        if type(self) is not type(other):
            return False
        return self._testMethodName == other._testMethodName

    def __hash__(self) -> int:
        return hash((type(self), self._testMethodName))

    def __str__(self) -> str:
        return "%s (%s.%s)" % (self._testMethodName, strclass(type(self)), self._testMethodName)

    def __repr__(self) -> str:
        return "<%s testMethod=%s>" % (strclass(type(self)), self._testMethodName)

    def defaultTestResult(self) -> TestResult:
        return TestResult()

    def _callSetUp(self) -> None:
        self.setUp()

    def _callTestMethod(self) -> None:
        getattr(self, self._testMethodName)()

    def _callTearDown(self) -> None:
        self.tearDown()

    def run(self, result: TestResult | None = None) -> TestResult:
        stopTestRun = False
        if result is None:
            result = self.defaultTestResult()
            result.startTestRun()
            stopTestRun = True
        result.startTest(self)
        try:
            outcome = _Outcome(result)
            start_time = time.perf_counter()
            try:
                self._outcome = outcome
                with outcome.testPartExecutor(self):
                    self._callSetUp()
                if outcome.success:
                    with outcome.testPartExecutor(self):
                        self._callTestMethod()
                    with outcome.testPartExecutor(self):
                        self._callTearDown()
                # WORKAROUND(ut_bug_exc_info_after_calling_a_callable_from_a_list):
                # cleanups are discarded. Calling one -- a callable taken off a
                # list -- in a function whose handlers read the traceback stops
                # the ownership verifier.
                self._cleanups.clear()
                result.addDuration(self, time.perf_counter() - start_time)
                if outcome.success:
                    expected = outcome.expectedFailure
                    if expected is not None:
                        result.addExpectedFailure(self, expected)
                    elif outcome.unexpectedSuccess:
                        result.addUnexpectedSuccess(self)
                    else:
                        result.addSuccess(self)
                return result
            finally:
                outcome.expectedFailure = None
                self._outcome = None
        finally:
            result.stopTest(self)
            if stopTestRun:
                result.stopTestRun()

    def __call__(self, result: TestResult | None = None) -> TestResult:
        return self.run(result)

    # WORKAROUND(ut_bug_class_body_field_object_into_inner_contextmanager):
    # subTest is not here -- its generator does not compile.

    def skipTest(self, reason: str) -> None:
        """Skip this test."""
        raise SkipTest(reason)

    def fail(self, msg: str | None = None) -> None:
        """Fail immediately, with the given message."""
        raise self.failureException(msg)

    def assertFalse(self, expr: object, msg: str | None = None) -> None:
        """Check that the expression is false."""
        if expr:
            self.fail(self._formatMessage(msg, "%s is not false" % safe_repr(expr)))

    def assertTrue(self, expr: object, msg: str | None = None) -> None:
        """Check that the expression is true."""
        if not expr:
            self.fail(self._formatMessage(msg, "%s is not true" % safe_repr(expr)))

    def _formatMessage(self, msg: str | None, standardMsg: str) -> str:
        if not self.longMessage:
            if msg is None:
                return standardMsg
            return msg
        if msg is None:
            return standardMsg
        return '%s : %s' % (standardMsg, msg)

    def assertRaises[E: BaseException](self, expected_exception: type[E],
                                       msg: str | None = None) -> _AssertRaisesContext[E]:
        """Fail unless an exception of class expected_exception is raised in
        the `with` block this returns the context manager for."""
        return _AssertRaisesContext[E](expected_exception, self, msg)

    def assertRaisesCall[E: BaseException, **P](self, expected_exception: type[E],
                                                callable_obj: Callable[P, object],
                                                *args: P.args, **kwargs: P.kwargs) -> None:
        """assertRaises' callable form: call callable_obj(*args, **kwargs)."""
        context = _AssertRaisesContext[E](expected_exception, self)
        with context:
            callable_obj(*args, **kwargs)

    def _baseAssertEqual[T](self, first: T, second: T, msg: str | None = None) -> None:
        """The default assertEqual implementation, not type specific."""
        if not first == second:
            standardMsg = '%s != %s' % _common_shorten_repr(safe_repr(first), safe_repr(second))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertEqual[T](self, first: T, second: T, msg: str | None = None) -> None:
        """Fail if the two objects are unequal as determined by the '=='
        operator."""
        if isinstance(first, str) and isinstance(second, str):
            self.assertMultiLineEqual(first, second, msg)
            return
        self._baseAssertEqual(first, second, msg)

    def assertNotEqual[T](self, first: T, second: T, msg: str | None = None) -> None:
        """Fail if the two objects are equal as determined by the '!='
        operator."""
        if not first != second:
            self.fail(self._formatMessage(
                msg, '%s == %s' % (safe_repr(first), safe_repr(second))))

    def assertAlmostEqual(self, first: float, second: float, places: int | None = None,
                          msg: str | None = None, delta: float | None = None) -> None:
        if first == second:
            return
        if delta is not None and places is not None:
            raise TypeError("specify delta or places not both")
        diff = abs(first - second)
        if delta is not None:
            if diff <= delta:
                return
            standardMsg = '%s != %s within %s delta (%s difference)' % (
                safe_repr(first), safe_repr(second), safe_repr(delta), safe_repr(diff))
        else:
            if places is None:
                places = 7
            if round(diff, places) == 0:
                return
            standardMsg = '%s != %s within %r places (%s difference)' % (
                safe_repr(first), safe_repr(second), places, safe_repr(diff))
        self.fail(self._formatMessage(msg, standardMsg))

    def assertSequenceEqual[T](self, seq1: list[T], seq2: list[T], msg: str | None = None,
                               seq_type_name: str = "sequence") -> None:
        if seq1 == seq2:
            return
        len1 = len(seq1)
        len2 = len(seq2)
        r1, r2 = _common_shorten_repr(safe_repr(seq1), safe_repr(seq2))
        differing = '%ss differ: %s != %s\n' % (seq_type_name.capitalize(), r1, r2)
        for i in range(min(len1, len2)):
            item1 = seq1[i]
            item2 = seq2[i]
            if item1 != item2:
                # WORKAROUND(ut_unsupported_fixed_tuple_concatenation): `fmt % ((i,) + pair)` is refused.
                e1, e2 = _common_shorten_repr(safe_repr(item1), safe_repr(item2))
                differing += '\nFirst differing element %d:\n%s\n%s\n' % (i, e1, e2)
                break
        if len1 > len2:
            differing += ('\nFirst %s contains %d additional '
                          'elements.\n' % (seq_type_name, len1 - len2))
            differing += ('First extra element %d:\n%s\n' % (len2, safe_repr(seq1[len2])))
        elif len1 < len2:
            differing += ('\nSecond %s contains %d additional '
                          'elements.\n' % (seq_type_name, len2 - len1))
            differing += ('First extra element %d:\n%s\n' % (len1, safe_repr(seq2[len1])))
        self.fail(self._formatMessage(msg, differing))

    def assertListEqual[T](self, list1: list[T], list2: list[T], msg: str | None = None) -> None:
        self.assertSequenceEqual(list1, list2, msg, "list")

    def assertDictEqual[K, V](self, d1: dict[K, V], d2: dict[K, V], msg: str | None = None) -> None:
        if d1 != d2:
            standardMsg = '%s != %s' % _common_shorten_repr(safe_repr(d1), safe_repr(d2))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertSetEqual[T](self, set1: set[T], set2: set[T], msg: str | None = None) -> None:
        difference1 = set1.difference(set2)
        difference2 = set2.difference(set1)
        if not (difference1 or difference2):
            return
        lines: list[str] = []
        if difference1:
            lines.append('Items in the first set but not the second:')
            for item in difference1:
                lines.append(repr(item))
        if difference2:
            lines.append('Items in the second set but not the first:')
            for item in difference2:
                lines.append(repr(item))
        self.fail(self._formatMessage(msg, '\n'.join(lines)))

    def assertIn[T](self, member: T, container: list[T], msg: str | None = None) -> None:
        """Just like self.assertTrue(a in b), but with a nicer default message."""
        if member not in container:
            standardMsg = '%s not found in %s' % (safe_repr(member), safe_repr(container))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertNotIn[T](self, member: T, container: list[T], msg: str | None = None) -> None:
        """Just like self.assertTrue(a not in b), but with a nicer default message."""
        if member in container:
            standardMsg = '%s unexpectedly found in %s' % (safe_repr(member), safe_repr(container))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIs(self, expr1: object, expr2: object, msg: str | None = None) -> None:
        """Just like self.assertTrue(a is b), but with a nicer default message."""
        if expr1 is not expr2:
            standardMsg = '%s is not %s' % (safe_repr(expr1), safe_repr(expr2))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIsNot(self, expr1: object, expr2: object, msg: str | None = None) -> None:
        """Just like self.assertTrue(a is not b), but with a nicer default message."""
        if expr1 is expr2:
            standardMsg = 'unexpectedly identical: %s' % (safe_repr(expr1),)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIsNone(self, obj: object, msg: str | None = None) -> None:
        """Same as self.assertTrue(obj is None), with a nicer default message."""
        if obj is not None:
            standardMsg = '%s is not None' % (safe_repr(obj),)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIsNotNone(self, obj: object, msg: str | None = None) -> None:
        """Included for symmetry with assertIsNone."""
        if obj is None:
            self.fail(self._formatMessage(msg, 'unexpectedly None'))

    def assertIsInstance(self, obj: object, cls: type, msg: str | None = None) -> None:
        """Same as self.assertTrue(isinstance(obj, cls)), with a nicer
        default message."""
        if not isinstance(obj, cls):
            standardMsg = f'{safe_repr(obj)} is not an instance of {cls!r}'
            self.fail(self._formatMessage(msg, standardMsg))

    def assertMultiLineEqual(self, first: str, second: str, msg: str | None = None) -> None:
        """Assert that two multi-line strings are equal."""
        if first != second:
            standardMsg = '%s != %s' % _common_shorten_repr(safe_repr(first), safe_repr(second))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertLess[T: float](self, a: T, b: T, msg: str | None = None) -> None:
        """Just like self.assertTrue(a < b), but with a nicer default message."""
        if not a < b:
            standardMsg = '%s not less than %s' % (safe_repr(a), safe_repr(b))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertGreater[T: float](self, a: T, b: T, msg: str | None = None) -> None:
        """Just like self.assertTrue(a > b), but with a nicer default message."""
        if not a > b:
            standardMsg = '%s not greater than %s' % (safe_repr(a), safe_repr(b))
            self.fail(self._formatMessage(msg, standardMsg))


class _SubTest(TestCase):

    def __init__(self, test_case: TestCase, message: str | None, params: dict[str, object]) -> None:
        super().__init__()
        self._message = message
        self.test_case = test_case
        self.params = params
        self.failureException = test_case.failureException

    def runTest(self) -> None:
        raise NotImplementedError("subtests cannot be run directly")

    def _subDescription(self) -> str:
        parts: list[str] = []
        message = self._message
        if message is not None:
            parts.append("[{}]".format(message))
        if self.params:
            params_desc = ', '.join("{}={!r}".format(k, v) for (k, v) in self.params.items())
            parts.append("({})".format(params_desc))
        text = " ".join(parts)
        if text == "":
            return '(<subtest>)'
        return text

    def id(self) -> str:
        return "{} {}".format(self.test_case.id(), self._subDescription())

    def __str__(self) -> str:
        return "{} {}".format(self.test_case, self._subDescription())


# ---------------------------------------------------------------- result

class TestResult:
    """Holder for test result information."""

    def __init__(self) -> None:
        self.failfast = False
        self.failures: list[tuple[TestCase, str]] = []
        self.errors: list[tuple[TestCase, str]] = []
        self.testsRun = 0
        self.skipped: list[tuple[TestCase, str]] = []
        self.expectedFailures: list[tuple[TestCase, str]] = []
        self.unexpectedSuccesses: list[TestCase] = []
        self.collectedDurations: list[tuple[str, float]] = []
        self.shouldStop = False
        self._previousTestClass: type[TestCase] | None = None
        self._classSetupFailed: dict[type[TestCase], bool] = {}

    def printErrors(self) -> None:
        "Called by TestRunner after test run"

    def startTest(self, test: TestCase) -> None:
        "Called when the given test is about to be run"
        self.testsRun += 1

    def stopTest(self, test: TestCase) -> None:
        """Called when the given test has been run"""

    def startTestRun(self) -> None:
        """Called once before any tests are executed."""

    def stopTestRun(self) -> None:
        """Called once after all tests are executed."""

    def addError(self, test: TestCase,
                 err: _SysExcInfoType) -> None:
        """Called when an error has occurred."""
        self.errors.append((test, _exc_info_to_string(err, test)))
        if self.failfast:
            self.stop()

    def addFailure(self, test: TestCase,
                   err: _SysExcInfoType) -> None:
        """Called when a failure has occurred."""
        self.failures.append((test, _exc_info_to_string(err, test)))
        if self.failfast:
            self.stop()

    def addSubTest(self, test: TestCase, subtest: TestCase,
                   err: _SysExcInfoType | None) -> None:
        """Called at the end of a subtest."""
        if err is not None:
            if self.failfast:
                self.stop()
            if issubclass(err[0], test.failureException):
                self.failures.append((subtest, _exc_info_to_string(err, test)))
            else:
                self.errors.append((subtest, _exc_info_to_string(err, test)))

    def addSuccess(self, test: TestCase) -> None:
        "Called when a test has completed successfully"
        pass

    def addSkip(self, test: TestCase, reason: str) -> None:
        """Called when a test is skipped."""
        self.skipped.append((test, reason))

    def addExpectedFailure(self, test: TestCase,
                           err: _SysExcInfoType) -> None:
        """Called when an expected failure/error occurred."""
        self.expectedFailures.append((test, _exc_info_to_string(err, test)))

    def addUnexpectedSuccess(self, test: TestCase) -> None:
        """Called when a test was expected to fail, but succeed."""
        self.unexpectedSuccesses.append(test)
        if self.failfast:
            self.stop()

    def addDuration(self, test: TestCase, elapsed: float) -> None:
        self.collectedDurations.append((str(test), elapsed))

    def wasSuccessful(self) -> bool:
        """Tells whether or not this result was a success."""
        return (len(self.failures) == 0 and len(self.errors) == 0
                and len(self.unexpectedSuccesses) == 0)

    def stop(self) -> None:
        """Indicates that the tests should be aborted."""
        self.shouldStop = True

    def __repr__(self) -> str:
        return ("<%s run=%i errors=%i failures=%i>" %
                (strclass(type(self)), self.testsRun, len(self.errors), len(self.failures)))


def _is_unittest_entry(entry: str) -> bool:
    first = entry.split("\n")[0]
    return 'unittest.py", line' in first


def _exc_info_to_string(err: _SysExcInfoType,
                        test: TestCase) -> str:
    """Converts a sys.exc_info()-style tuple of values into a string."""
    exctype, value, tb = err
    entries = traceback.format_tb(tb)
    # Skip test runner traceback levels, then (for a failure) the assert*
    # levels below the test's own frame.
    start = 0
    while start < len(entries) and _is_unittest_entry(entries[start]):
        start += 1
    end = len(entries)
    if exctype is test.failureException:
        cut = start
        while cut < end and not _is_unittest_entry(entries[cut]):
            cut += 1
        end = cut
    msgLines: list[str] = []
    if start < end:
        msgLines.append('Traceback (most recent call last):\n')
        for entry in entries[start:end]:
            msgLines.append(entry)
    for line in traceback.format_exception_only(value):
        msgLines.append(line)
    return ''.join(msgLines)


# ---------------------------------------------------------------- suite

class TestSuite:
    """A test suite is a composite test consisting of a number of TestCases."""

    def __init__(self, tests: list[TestCase | TestSuite] | None = None) -> None:
        self._tests: list[TestCase | TestSuite] = []
        if tests is not None:
            for test in tests:
                self.addTest(test)

    def __repr__(self) -> str:
        return "<%s tests=%s>" % (strclass(type(self)), self._tests)

    def countTestCases(self) -> int:
        cases = 0
        for test in self._tests:
            cases += test.countTestCases()
        return cases

    def addTest(self, test: TestCase | TestSuite) -> None:
        self._tests.append(test)

    def run(self, result: TestResult) -> TestResult:
        _run_suite(self, result, True)
        return result


def _run_suite(suite: TestSuite, result: TestResult, topLevel: bool) -> None:
    for test in suite._tests:
        if result.shouldStop:
            break
        if isinstance(test, TestCase):
            _tearDownPreviousClass(test, result)
            _handleClassSetUp(test, result)
            result._previousTestClass = type(test)
            if result._classSetupFailed.get(type(test), False):
                continue
            test.run(result)
        else:
            _run_suite(test, result, False)
    if topLevel:
        _tearDownPreviousClass(None, result)


def _handleClassSetUp(test: TestCase, result: TestResult) -> None:
    currentClass = type(test)
    if currentClass is result._previousTestClass:
        return
    result._classSetupFailed[currentClass] = False
    try:
        currentClass.setUpClass()
    except Exception as e:
        result._classSetupFailed[currentClass] = True
        # WORKAROUND(ut_unsupported_exc_info_outside_handler): sys.exc_info() is answered only lexically
        # inside an except block, so the handler passes it down.
        _addClassLevelException(result, e, 'setUpClass (%s)' % strclass(currentClass),
                                sys.exc_info())


def _tearDownPreviousClass(test: TestCase | None, result: TestResult) -> None:
    previousClass = result._previousTestClass
    if previousClass is None:
        return
    if test is not None and type(test) is previousClass:
        return
    if result._classSetupFailed.get(previousClass, False):
        return
    try:
        previousClass.tearDownClass()
    except Exception as e:
        _addClassLevelException(result, e, 'tearDownClass (%s)' % strclass(previousClass),
                                sys.exc_info())


def _addClassLevelException(result: TestResult, exception: Exception, errorName: str,
                            info: _SysExcInfoType) -> None:
    error = _ErrorHolder(errorName)
    if isinstance(exception, SkipTest):
        result.addSkip(error, str(exception))
    else:
        result.addError(error, info)


class _ErrorHolder(TestCase):
    """Placeholder for a TestCase inside a result."""

    def __init__(self, description: str) -> None:
        super().__init__()
        self.description = description

    def runTest(self) -> None:
        pass

    def id(self) -> str:
        return self.description

    def __repr__(self) -> str:
        return "<ErrorHolder description=%r>" % (self.description,)

    def __str__(self) -> str:
        return self.id()

    def countTestCases(self) -> int:
        return 0


# ---------------------------------------------------------------- loader

class TestLoader:
    """Loads the test methods of TestCase classes into suites."""

    testMethodPrefix = 'test'

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.testNamePatterns: list[str] | None = None

    def getTestCaseNames(self, testCaseClass: type[TestCase]) -> list[str]:
        """Return a sorted sequence of method names found within testCaseClass."""
        names: list[str] = []
        for attrname in dir(testCaseClass):
            if not attrname.startswith(self.testMethodPrefix):
                continue
            patterns = self.testNamePatterns
            if patterns is not None:
                fullName = '%s.%s.%s' % (testCaseClass.__module__, testCaseClass.__qualname__, attrname)
                if not any(p in fullName for p in patterns):
                    continue
            names.append(attrname)
        names.sort()
        return names

    def loadTestsFromTestCase(self, testCaseClass: type[TestCase]) -> TestSuite:
        """Return a suite of all test cases contained in testCaseClass."""
        testCaseNames = self.getTestCaseNames(testCaseClass)
        if not testCaseNames and hasattr(testCaseClass, 'runTest'):
            testCaseNames = ['runTest']
        tests: list[TestCase | TestSuite] = []
        for name in testCaseNames:
            tests.append(testCaseClass(name))
        return TestSuite(tests)


# ---------------------------------------------------------------- runner

class _WritelnDecorator:
    """Used to decorate file-like objects with a handy 'writeln' method."""

    def __init__(self, toStdout: bool) -> None:
        self.toStdout = toStdout

    def write(self, arg: str) -> None:
        if self.toStdout:
            sys.stdout.write(arg)
        else:
            sys.stderr.write(arg)

    def flush(self) -> None:
        if self.toStdout:
            sys.stdout.flush()
        else:
            sys.stderr.flush()

    def writeln(self, arg: str | None = None) -> None:
        if arg:
            self.write(arg)
        self.write('\n')


class TextTestResult(TestResult):
    """A test result class that can print formatted text results to a stream."""

    separator1 = '=' * 70
    separator2 = '-' * 70

    def __init__(self, stream: _WritelnDecorator, descriptions: bool, verbosity: int) -> None:
        super().__init__()
        self.stream = stream
        self.showAll = verbosity > 1
        self.dots = verbosity == 1
        self.descriptions = descriptions
        self._newline = True

    def getDescription(self, test: TestCase) -> str:
        doc_first_line = test.shortDescription()
        if self.descriptions and doc_first_line:
            # WORKAROUND(ut_bug_join_of_tuple_with_narrowed_optional): not
            # '\n'.join((str(test), doc_first_line)).
            return str(test) + '\n' + doc_first_line
        return str(test)

    def startTest(self, test: TestCase) -> None:
        super().startTest(test)
        if self.showAll:
            self.stream.write(self.getDescription(test))
            self.stream.write(" ... ")
            self.stream.flush()
            self._newline = False

    def _write_status(self, test: TestCase, status: str) -> None:
        is_subtest = isinstance(test, _SubTest)
        if is_subtest or self._newline:
            if not self._newline:
                self.stream.writeln()
            if is_subtest:
                self.stream.write("  ")
            self.stream.write(self.getDescription(test))
            self.stream.write(" ... ")
        self.stream.writeln(status)
        self.stream.flush()
        self._newline = True

    def addSubTest(self, test: TestCase, subtest: TestCase,
                   err: _SysExcInfoType | None) -> None:
        if err is not None:
            if self.showAll:
                if issubclass(err[0], subtest.failureException):
                    self._write_status(subtest, "FAIL")
                else:
                    self._write_status(subtest, "ERROR")
            elif self.dots:
                if issubclass(err[0], subtest.failureException):
                    self.stream.write("F")
                else:
                    self.stream.write("E")
                self.stream.flush()
        super().addSubTest(test, subtest, err)

    def addSuccess(self, test: TestCase) -> None:
        super().addSuccess(test)
        if self.showAll:
            self._write_status(test, "ok")
        elif self.dots:
            self.stream.write(".")
            self.stream.flush()

    def addError(self, test: TestCase,
                 err: _SysExcInfoType) -> None:
        super().addError(test, err)
        if self.showAll:
            self._write_status(test, "ERROR")
        elif self.dots:
            self.stream.write("E")
            self.stream.flush()

    def addFailure(self, test: TestCase,
                   err: _SysExcInfoType) -> None:
        super().addFailure(test, err)
        if self.showAll:
            self._write_status(test, "FAIL")
        elif self.dots:
            self.stream.write("F")
            self.stream.flush()

    def addSkip(self, test: TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        if self.showAll:
            self._write_status(test, "skipped {!r}".format(reason))
        elif self.dots:
            self.stream.write("s")
            self.stream.flush()

    def addExpectedFailure(self, test: TestCase,
                           err: _SysExcInfoType) -> None:
        super().addExpectedFailure(test, err)
        if self.showAll:
            self.stream.writeln("expected failure")
            self.stream.flush()
        elif self.dots:
            self.stream.write("x")
            self.stream.flush()

    def addUnexpectedSuccess(self, test: TestCase) -> None:
        super().addUnexpectedSuccess(test)
        if self.showAll:
            self.stream.writeln("unexpected success")
            self.stream.flush()
        elif self.dots:
            self.stream.write("u")
            self.stream.flush()

    def printErrors(self) -> None:
        if self.dots or self.showAll:
            self.stream.writeln()
            self.stream.flush()
        self.printErrorList("ERROR", self.errors)
        self.printErrorList("FAIL", self.failures)
        if self.unexpectedSuccesses:
            self.stream.writeln(self.separator1)
            for test in self.unexpectedSuccesses:
                self.stream.writeln("UNEXPECTED SUCCESS: " + self.getDescription(test))
            self.stream.flush()

    def printErrorList(self, flavour: str, errors: list[tuple[TestCase, str]]) -> None:
        for test, err in errors:
            self.stream.writeln(self.separator1)
            self.stream.writeln("%s: %s" % (flavour, self.getDescription(test)))
            self.stream.writeln(self.separator2)
            self.stream.writeln("%s" % err)
            self.stream.flush()


class TextTestRunner:
    """A test runner class that displays results in textual form."""

    def __init__(self, toStdout: bool = False, descriptions: bool = True, verbosity: int = 1,
                 failfast: bool = False) -> None:
        self.stream = _WritelnDecorator(toStdout)
        self.descriptions = descriptions
        self.verbosity = verbosity
        self.failfast = failfast

    def run(self, test: TestSuite) -> TextTestResult:
        "Run the given test case or test suite."
        result = TextTestResult(self.stream, self.descriptions, self.verbosity)
        result.failfast = self.failfast
        start_time = time.perf_counter()
        test.run(result)
        stop_time = time.perf_counter()
        time_taken = stop_time - start_time
        result.printErrors()
        self.stream.writeln(result.separator2)
        run = result.testsRun
        self.stream.writeln("Ran %d test%s in %.3fs" %
                            (run, run != 1 and "s" or "", time_taken))
        self.stream.writeln()
        expected_fails = len(result.expectedFailures)
        unexpected_successes = len(result.unexpectedSuccesses)
        skipped = len(result.skipped)
        infos: list[str] = []
        if not result.wasSuccessful():
            self.stream.write("FAILED")
            failed, errored = len(result.failures), len(result.errors)
            if failed != 0:
                infos.append("failures=%d" % failed)
            if errored != 0:
                infos.append("errors=%d" % errored)
        elif run == 0 and skipped == 0:
            self.stream.write("NO TESTS RAN")
        else:
            self.stream.write("OK")
        if skipped != 0:
            infos.append("skipped=%d" % skipped)
        if expected_fails != 0:
            infos.append("expected failures=%d" % expected_fails)
        if unexpected_successes != 0:
            infos.append("unexpected successes=%d" % unexpected_successes)
        if infos:
            self.stream.writeln(" (%s)" % (", ".join(infos),))
        else:
            self.stream.write("\n")
        self.stream.flush()
        return result


# ---------------------------------------------------------------- main

_NO_TESTS_EXITCODE: int = 5


def main(*testCaseClasses: type[TestCase]) -> None:
    """Run the test methods of the given classes; exit with the outcome."""
    verbosity = 1
    failfast = False
    patterns: list[str] = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '-v' or arg == '--verbose':
            verbosity = 2
        elif arg == '-q' or arg == '--quiet':
            verbosity = 0
        elif arg == '-f' or arg == '--failfast':
            failfast = True
        elif arg == '-k' and i + 1 < len(argv):
            i += 1
            patterns.append(argv[i])
        i += 1
    loader = TestLoader()
    if patterns:
        loader.testNamePatterns = patterns
    suite = TestSuite()
    for cls in testCaseClasses:
        suite.addTest(loader.loadTestsFromTestCase(cls))
    runner = TextTestRunner(verbosity=verbosity, failfast=failfast)
    result = runner.run(suite)
    if result.testsRun == 0 and len(result.skipped) == 0:
        sys.exit(_NO_TESTS_EXITCODE)
    sys.exit(0 if result.wasSuccessful() else 1)
