"""Test case implementation"""

import sys
import difflib
import pprint
import re
import contextlib
import warnings
import logging
import time
import types
from collections.abc import Callable, Iterator
from typing import overload

from . import result
from .util import strclass, safe_repr, _common_shorten_repr

__unittest = True

_SysExcInfoType = tuple[type[BaseException], BaseException, types.TracebackType]

DIFF_OMITTED = ('\nDiff is %s characters long. '
                 'Set self.maxDiff to None to see it.')

_subtest_msg_sentinel = object()

class SkipTest(Exception):
    """
    Raise this exception in a test to skip it.

    Usually you can use TestCase.skipTest() or one of the skipping decorators
    instead of raising this directly.
    """

class _ShouldStop(Exception):
    """
    The test should stop.
    """

class _UnexpectedSuccess(Exception):
    """
    The test was supposed to fail, but it didn't!
    """

class _ExpectedFailure(Exception):
    """
    The test failed as its decorator said it would.
    """

    def __init__(self, exc_info: _SysExcInfoType) -> None:
        super().__init__()
        self.exc_info = exc_info


class _Outcome(object):
    def __init__(self, result: result.TestResult | None = None) -> None:
        self.expecting_failure = False
        self.result = result
        self.result_supports_subtests = result is not None
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
            _addSkip(self.result, test_case, str(e))
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
                result = self.result
                if subTest and result is not None and isinstance(test_case, _SubTest):
                    result.addSubTest(test_case.test_case, test_case, exc_info)
                else:
                    _addError(self.result, test_case, exc_info)
        else:
            result = self.result
            if subTest and self.success and result is not None and isinstance(test_case, _SubTest):
                result.addSubTest(test_case.test_case, test_case, None)
        finally:
            self.success = self.success and old_success


def _addSkip(result: result.TestResult | None, test_case: TestCase, reason: str) -> None:
    if result is not None:
        result.addSkip(test_case, reason)

def _addError(result: result.TestResult | None, test: TestCase, exc_info: _SysExcInfoType) -> None:
    if result is not None and exc_info is not None:
        if issubclass(exc_info[0], test.failureException):
            result.addFailure(test, exc_info)
        else:
            result.addError(test, exc_info)

def _id[T](obj: T) -> T:
    return obj


_module_cleanups: list[Callable[[], object]] = []
def addModuleCleanup[**P](function: Callable[P, object], /, *args: P.args, **kwargs: P.kwargs) -> None:
    """Same as addCleanup, except the cleanup items are called even if
    setUpModule fails (unlike tearDownModule)."""
    _module_cleanups.append(lambda: function(*args, **kwargs))


def doModuleCleanups() -> None:
    """Execute all module cleanup functions. Normally called for you after
    tearDownModule."""
    exceptions: list[Exception] = []
    while _module_cleanups:
        function = _module_cleanups.pop()
        try:
            function()
        except Exception as exc:
            exceptions.append(exc)
    if exceptions:
        # Swallows all but first exception. If a multi-exception handler
        # gets written we should use that here instead.
        raise exceptions[0]


# The skipped classes, and why: what CPython keeps as `__unittest_skip__` and
# `__unittest_skip_why__` attributes of the class, found here through
# issubclass so that a subclass is skipped too, as it inherits them there.
_skipped_classes: list[tuple[type[TestCase], str]] = []


def _class_skip_reason(cls: type[TestCase]) -> str | None:
    for skipped, why in _skipped_classes:
        if issubclass(cls, skipped):
            return why
    return None


class _SkipDecorator:
    """What skip / skipIf / skipUnless return: it decorates a test method
    (`__call__`) or a TestCase class (`__ly_decorate_class__`)."""

    def __init__(self, reason: str, active: bool) -> None:
        self.reason = reason
        self.active = active

    def __call__[T: TestCase](self, test_item: Callable[[T], None]) -> Callable[[T], None]:
        if not self.active:
            return test_item
        reason = self.reason
        def skip_wrapper(self: T) -> None:
            raise SkipTest(reason)
        return skip_wrapper

    def __ly_decorate_class__[C: TestCase](self, test_item: type[C]) -> type[C]:
        if self.active:
            _skipped_classes.append((test_item, self.reason))
        return test_item


def skip(reason: str) -> _SkipDecorator:
    """
    Unconditionally skip a test.
    """
    return _SkipDecorator(reason, True)

def skipIf(condition: bool, reason: str) -> _SkipDecorator:
    """
    Skip a test if the condition is true.
    """
    return _SkipDecorator(reason, condition)

def skipUnless(condition: bool, reason: str) -> _SkipDecorator:
    """
    Skip a test unless the condition is true.
    """
    return _SkipDecorator(reason, not condition)

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

    def __init__(self, expected: type[E] | tuple[type[E], ...],
                 test_case: TestCase, msg: str | None = None) -> None:
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
        expected = self.expected
        if exc_type is None:
            # ⛔ Not CPython's `__name__` with AttributeError caught: the
            # union says which of the two it is.
            if isinstance(expected, tuple):
                exc_name = str(expected)
            else:
                exc_name = expected.__name__
            obj_name = self.obj_name
            if obj_name is not None:
                self.test_case.fail(self.test_case._formatMessage(
                    self.msg, "{} not raised by {}".format(exc_name, obj_name)))
            else:
                self.test_case.fail(self.test_case._formatMessage(
                    self.msg, "{} not raised".format(exc_name)))
            return False
        # ⛔ Not `issubclass(exc_type, self.expected)`: issubclass takes a
        # class or a tuple here, not a union of the two.
        classes = expected if isinstance(expected, tuple) else (expected,)
        matched = False
        for cls in classes:
            if issubclass(exc_type, cls):
                matched = True
        if not matched:
            # let unexpected exceptions pass through
            return False
        for cls in classes:
            if isinstance(exc_value, cls):
                self._exception = exc_value
        return True


class _AssertRaisesRegexContext[E: BaseException]:
    """assertRaisesRegex's context manager: _AssertRaisesContext's, and the
    raised exception's text searched for expected_regex."""

    def __init__(self, expected: type[E] | tuple[type[E], ...],
                 test_case: TestCase, expected_regex: str | re.Pattern,
                 msg: str | None = None) -> None:
        self.raises = _AssertRaisesContext[E](expected, test_case, msg)
        self.expected_regex = re.compile(expected_regex)

    @property
    def exception(self) -> E:
        return self.raises.exception

    def __enter__(self) -> _AssertRaisesRegexContext[E]:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None,
                 tb: types.TracebackType | None) -> bool:
        # ⛔ Not `tb`: a traceback has no value here, and the inner context
        # does not read it.
        if not self.raises.__exit__(exc_type, exc_value, None):
            return False
        expected_regex = self.expected_regex
        if expected_regex.search(str(exc_value)) is None:
            test_case = self.raises.test_case
            test_case.fail(test_case._formatMessage(
                self.raises.msg, '"{}" does not match "{}"'.format(
                    expected_regex.pattern, str(exc_value))))
        return True


class _AssertWarnsContext[W: Warning]:
    """A context manager used to implement TestCase.assertWarns* methods."""

    def __init__(self, expected: type[W], test_case: TestCase,
                 expected_regex: str | re.Pattern | None = None,
                 msg: str | None = None) -> None:
        self.expected = expected
        self.test_case = test_case
        self.msg = msg
        self.obj_name: str | None = None
        self.expected_regex: re.Pattern | None = None
        if expected_regex is not None:
            self.expected_regex = re.compile(expected_regex)
        self.warnings_manager = warnings.catch_warnings(record=True)
        self.warnings: list[warnings.WarningMessage] = []
        self._warning: W | None = None
        self.filename = ""
        self.lineno = 0

    @property
    def warning(self) -> W:
        w = self._warning
        if w is None:
            raise AttributeError("'_AssertWarnsContext' object has no attribute 'warning'")
        return w

    def _raiseFailure(self, standardMsg: str) -> None:
        self.test_case.fail(self.test_case._formatMessage(self.msg, standardMsg))

    def __enter__(self) -> _AssertWarnsContext[W]:
        # The __warningregistry__'s need to be in a pristine state for tests
        # to work properly.
        warnings._clear_registries()
        self.warnings = self.warnings_manager._enter_recording()
        warnings.simplefilter("always", self.expected)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None,
                 tb: types.TracebackType | None) -> None:
        self.warnings_manager.__exit__(exc_type, exc_value, None)
        if exc_type is not None:
            # let unexpected exceptions pass through
            return
        exc_name = self.expected.__name__
        first_matching: Warning | None = None
        for m in self.warnings:
            w = m.message
            if not isinstance(w, self.expected):
                continue
            if first_matching is None:
                first_matching = w
            expected_regex = self.expected_regex
            if (expected_regex is not None and
                expected_regex.search(str(w)) is None):
                continue
            # store warning for later retrieval
            self._warning = w
            self.filename = m.filename
            self.lineno = m.lineno
            return
        # Now we simply try to choose a helpful failure message
        expected_regex = self.expected_regex
        if first_matching is not None and expected_regex is not None:
            self._raiseFailure('"{}" does not match "{}"'.format(
                     expected_regex.pattern, str(first_matching)))
        obj_name = self.obj_name
        if obj_name is not None:
            self._raiseFailure("{} not triggered by {}".format(exc_name,
                                                               obj_name))
        else:
            self._raiseFailure("{} not triggered".format(exc_name))


class _LoggingWatcher:
    """The records and their formatted output an assertLogs block captured
    (CPython's is a namedtuple of the two lists)."""

    def __init__(self) -> None:
        self.records: list[logging.LogRecord] = []
        self.output: list[str] = []


class _CapturingHandler(logging.Handler):
    """
    A logging handler capturing all (raw and formatted) logging output.
    """

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        self.watcher = _LoggingWatcher()

    def flush(self) -> None:
        pass

    def emit(self, record: logging.LogRecord) -> None:
        self.watcher.records.append(record)
        msg = self.format(record)
        self.watcher.output.append(msg)


class _LogsContext:
    """What assertLogs and assertNoLogs share: the logger's handlers swapped
    for a capturing one on entry and put back on exit."""

    LOGGING_FORMAT = "%(levelname)s:%(name)s:%(message)s"

    def __init__(self, test_case: TestCase,
                 logger_name: str | logging.Logger | None,
                 level: int | str | None) -> None:
        self.test_case = test_case
        self.logger_name = logger_name
        self.level = logging.INFO
        if isinstance(level, str):
            self.level = logging._checkLevel(level)
        elif level is not None and level != 0:
            self.level = level
        self.msg: str | None = None
        self.logger: logging.Logger = logging.root
        self.watcher = _LoggingWatcher()
        self.old_handlers: list[logging.Handler] = []
        self.old_level = 0
        self.old_propagate = True

    def _raiseFailure(self, standardMsg: str) -> None:
        self.test_case.fail(self.test_case._formatMessage(self.msg, standardMsg))

    def _start(self) -> None:
        logger_name = self.logger_name
        if isinstance(logger_name, logging.Logger):
            logger = self.logger = logger_name
        else:
            logger = self.logger = logging.getLogger(logger_name)
        formatter = logging.Formatter(self.LOGGING_FORMAT)
        handler = _CapturingHandler()
        handler.setLevel(self.level)
        handler.setFormatter(formatter)
        self.watcher = handler.watcher
        self.old_handlers = logger.handlers[:]
        self.old_level = logger.level
        self.old_propagate = logger.propagate
        logger.handlers = [handler]
        logger.setLevel(self.level)
        logger.propagate = False

    def _restore(self) -> None:
        self.logger.handlers = self.old_handlers
        self.logger.propagate = self.old_propagate
        self.logger.setLevel(self.old_level)


class _AssertLogsContext(_LogsContext):
    """A context manager for assertLogs()"""

    def __enter__(self) -> _LoggingWatcher:
        self._start()
        return self.watcher

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None,
                 tb: types.TracebackType | None) -> bool:
        self._restore()
        if exc_type is not None:
            # let unexpected exceptions pass through
            return False
        if len(self.watcher.records) == 0:
            self._raiseFailure(
                "no logs of level {} or higher triggered on {}"
                .format(logging.getLevelName(self.level), self.logger.name))
        return False


class _AssertNoLogsContext(_LogsContext):
    """A context manager for assertNoLogs()"""

    def __enter__(self) -> None:
        self._start()

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None,
                 tb: types.TracebackType | None) -> bool:
        self._restore()
        if exc_type is not None:
            # let unexpected exceptions pass through
            return False
        if len(self.watcher.records) > 0:
            self._raiseFailure(
                "Unexpected logs found: {!r}".format(
                    self.watcher.output
                )
            )
        return False


def _new_child(parent: dict[str, object], params: dict[str, object]) -> dict[str, object]:
    """_OrderedChainMap.new_child's view, flattened: the child's keys first,
    then the parent's that the child does not shadow."""
    merged = dict(params)
    for key, value in parent.items():
        if key not in merged:
            merged[key] = value
    return merged


class TestCase(object):
    """A class whose instances are single test cases.

    By default, the test code itself should be placed in a method named
    'runTest'.

    If the fixture may be used for many test cases, create as
    many test methods as are needed. When instantiating such a TestCase
    subclass, specify in the constructor arguments the name of the test method
    that the instance is to execute.

    Test authors should subclass TestCase for their own tests. Construction
    and deconstruction of the test's environment ('fixture') can be
    implemented by overriding the 'setUp' and 'tearDown' methods respectively.

    If it is necessary to override the __init__ method, the base class
    __init__ method must always be called. It is important that subclasses
    should not change the signature of their __init__ method, since instances
    of the classes are instantiated automatically by parts of the framework
    in order to be run.

    When subclassing TestCase, you can set these attributes:
    * failureException: determines which exception will be raised when
        the instance's assertion methods fail; test methods raising this
        exception will be deemed to have 'failed' rather than 'errored'.
    * longMessage: determines whether long messages (including repr of
        objects used in assert methods) will be printed on failure in *addition*
        to any explicit message passed.
    * maxDiff: sets the maximum length of a diff in failure messages
        by assert methods using difflib. It is looked up as an instance
        attribute so can be configured by individual tests if required.
    """

    failureException: type[BaseException] = AssertionError

    longMessage = True

    maxDiff: int | None = 80*8

    # If a string is longer than _diffThreshold, use normal comparison instead
    # of difflib.  See #11763.
    _diffThreshold = 2**16

    def __init__(self, methodName: str = 'runTest') -> None:
        """Create an instance of the class that will use the named test
           method when executed. Raises a ValueError if the instance does
           not have a method with the specified name.
        """
        self._testMethodName = methodName
        self._outcome: _Outcome | None = None
        self._testMethodDoc: str | None = 'No test'
        try:
            self._testMethodDoc = getattr(self, methodName).__doc__
        except AttributeError:
            if methodName != 'runTest':
                # we allow instantiation with no explicit method name
                # but not an *incorrect* or missing method name
                raise ValueError("no such test method in %s: %s" %
                      (self.__class__, methodName))
        self._cleanups: list[Callable[[], object]] = []
        self._subtest: _SubTest | None = None

    def addCleanup[**P](self, function: Callable[P, object], /, *args: P.args, **kwargs: P.kwargs) -> None:
        """Add a function, with arguments, to be called when the test is
        completed. Functions added are called on a LIFO basis and are
        called after tearDown on test failure or success.

        Cleanup items are called even if setUp fails (unlike tearDown)."""
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

    def defaultTestResult(self) -> result.TestResult:
        return result.TestResult()

    def shortDescription(self) -> str | None:
        """Returns a one-line description of the test, or None if no
        description has been provided.

        The default implementation of this method returns the first line of
        the specified test method's docstring.
        """
        doc = self._testMethodDoc
        return doc.strip().split("\n")[0].strip() if doc else None


    def id(self) -> str:
        return "%s.%s" % (strclass(self.__class__), self._testMethodName)

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented

        return self._testMethodName == other._testMethodName

    def __hash__(self) -> int:
        return hash((type(self), self._testMethodName))

    def __str__(self) -> str:
        return "%s (%s.%s)" % (self._testMethodName, strclass(self.__class__), self._testMethodName)

    def __repr__(self) -> str:
        return "<%s testMethod=%s>" % \
               (strclass(self.__class__), self._testMethodName)

    @contextlib.contextmanager
    def subTest(self, msg: object = _subtest_msg_sentinel, **params: object) -> Iterator[None]:
        """Return a context manager that will return the enclosed block
        of code in a subtest identified by the optional message and
        keyword parameters.  A failure in the subtest marks the test
        case as failed but resumes execution at the end of the enclosed
        block, allowing further test code to be executed.
        """
        if self._outcome is None or not self._outcome.result_supports_subtests:
            yield
            return
        parent = self._subtest
        if parent is None:
            params_map = dict(params)
        else:
            params_map = _new_child(parent.params, params)
        self._subtest = _SubTest(self, msg, params_map)
        try:
            with self._outcome.testPartExecutor(self._subtest, subTest=True):
                yield
            if not self._outcome.success:
                result = self._outcome.result
                if result is not None and result.failfast:
                    raise _ShouldStop
            elif self._outcome.expectedFailure:
                # If the test is expecting a failure, we really want to
                # stop now and register the expected failure.
                raise _ShouldStop
        finally:
            self._subtest = parent

    def _callSetUp(self) -> None:
        self.setUp()

    def _callTestMethod(self) -> None:
        getattr(self, self._testMethodName)()

    def _callTearDown(self) -> None:
        self.tearDown()

    def _callCleanup(self, function: Callable[[], object]) -> None:
        function()

    def run(self, result: result.TestResult | None = None) -> result.TestResult:
        if result is None:
            result = self.defaultTestResult()
            startTestRun = getattr(result, 'startTestRun', None)
            stopTestRun = getattr(result, 'stopTestRun', None)
            if startTestRun is not None:
                startTestRun()
        else:
            stopTestRun = None

        result.startTest(self)
        class_skip = _class_skip_reason(type(self))
        if class_skip is not None:
            # If the class was skipped.
            try:
                _addSkip(result, self, class_skip)
            finally:
                result.stopTest(self)
                if stopTestRun is not None:
                    stopTestRun()
            return result
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
                self.doCleanups()
                result.addDuration(self, (time.perf_counter() - start_time))

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
                # explicitly break reference cycle:
                # outcome.expectedFailure -> frame -> outcome -> outcome.expectedFailure
                outcome.expectedFailure = None
                outcome = None

                # clear the outcome, no more needed
                self._outcome = None

        finally:
            result.stopTest(self)
            if stopTestRun is not None:
                stopTestRun()

    def doCleanups(self) -> bool:
        """Execute all cleanup functions. Normally called for you after
        tearDown."""
        outcome = self._outcome or _Outcome()
        while self._cleanups:
            function = self._cleanups.pop()
            with outcome.testPartExecutor(self):
                self._callCleanup(function)

        # return this for backwards compatibility
        # even though we no longer use it internally
        return outcome.success

    def __call__(self, result: result.TestResult | None = None) -> result.TestResult:
        return self.run(result)

    def skipTest(self, reason: str) -> None:
        """Skip this test."""
        raise SkipTest(reason)

    def fail(self, msg: str | None = None) -> None:
        """Fail immediately, with the given message."""
        raise self.failureException(msg)

    def assertFalse(self, expr: object, msg: str | None = None) -> None:
        """Check that the expression is false."""
        if expr:
            msg = self._formatMessage(msg, "%s is not false" % safe_repr(expr))
            raise self.failureException(msg)

    def assertTrue(self, expr: object, msg: str | None = None) -> None:
        """Check that the expression is true."""
        if not expr:
            msg = self._formatMessage(msg, "%s is not true" % safe_repr(expr))
            raise self.failureException(msg)

    def _formatMessage(self, msg: str | None, standardMsg: str) -> str:
        """Honour the longMessage attribute when generating failure messages.
        If longMessage is False this means:
        * Use only an explicit message if it is provided
        * Otherwise use the standard message for the assert

        If longMessage is True:
        * Use the standard message
        * If an explicit message is provided, plus ' : ' and the explicit message
        """
        if not self.longMessage:
            return msg or standardMsg
        if msg is None:
            return standardMsg
        try:
            # don't switch to '{}' formatting in Python 2.X
            # it changes the way unicode input is handled
            return '%s : %s' % (standardMsg, msg)
        except UnicodeDecodeError:
            return  '%s : %s' % (safe_repr(standardMsg), safe_repr(msg))

    @overload
    def assertRaises[**P](self, expected_exception: type[BaseException] | tuple[type[BaseException], ...],
                          callable: Callable[P, object], /,
                          *args: P.args, **kwargs: P.kwargs) -> None: ...
    @overload
    def assertRaises[E: BaseException](self, expected_exception: type[E] | tuple[type[E], ...], *,
                                       msg: str | None = None) -> _AssertRaisesContext[E]: ...
    def assertRaises[E: BaseException, **P](self, expected_exception: type[E] | tuple[type[E], ...],
                                            callable_obj: Callable[P, object] | None = None,
                                            *args: P.args, msg: str | None = None,
                                            **kwargs: P.kwargs) -> _AssertRaisesContext[E] | None:
        """Fail unless an exception of class expected_exception is raised
           by the callable when invoked with specified positional and
           keyword arguments. If a different type of exception is
           raised, it will not be caught, and the test case will be
           deemed to have suffered an error, exactly as for an
           unexpected exception.

           If called with the callable and arguments omitted, will return a
           context object used like this::

                with self.assertRaises(SomeException):
                    do_something()

           An optional keyword argument 'msg' can be provided when assertRaises
           is used as a context object.

           The context manager keeps a reference to the exception as
           the 'exception' attribute. This allows you to inspect the
           exception after the assertion::

               with self.assertRaises(SomeException) as cm:
                   do_something()
               the_exception = cm.exception
               self.assertEqual(the_exception.error_code, 3)
        """
        context = _AssertRaisesContext[E](expected_exception, self, msg)
        if callable_obj is None:
            return context
        context.obj_name = callable_obj.__name__
        with context:
            callable_obj(*args, **kwargs)
        return None

    @overload
    def assertRaisesRegex[**P](self, expected_exception: type[BaseException] | tuple[type[BaseException], ...],
                               expected_regex: str | re.Pattern,
                               callable: Callable[P, object], /,
                               *args: P.args, **kwargs: P.kwargs) -> None: ...
    @overload
    def assertRaisesRegex[E: BaseException](self, expected_exception: type[E] | tuple[type[E], ...],
                                            expected_regex: str | re.Pattern, *,
                                            msg: str | None = None) -> _AssertRaisesRegexContext[E]: ...
    def assertRaisesRegex[E: BaseException, **P](self, expected_exception: type[E] | tuple[type[E], ...],
                                                 expected_regex: str | re.Pattern,
                                                 callable_obj: Callable[P, object] | None = None,
                                                 *args: P.args, msg: str | None = None,
                                                 **kwargs: P.kwargs) -> _AssertRaisesRegexContext[E] | None:
        """Asserts that the message in a raised exception matches a regex.

        Args:
            expected_exception: Exception class expected to be raised.
            expected_regex: Regex (re.Pattern object or string) expected
                    to be found in error message.
            args: Function to be called and extra positional args.
            kwargs: Extra kwargs.
            msg: Optional message used in case of failure. Can only be used
                    when assertRaisesRegex is used as a context manager.
        """
        context = _AssertRaisesRegexContext[E](expected_exception, self, expected_regex, msg)
        if callable_obj is None:
            return context
        context.raises.obj_name = callable_obj.__name__
        with context:
            callable_obj(*args, **kwargs)
        return None

    @overload
    def assertWarns[**P](self, expected_warning: type[Warning],
                         callable: Callable[P, object], /,
                         *args: P.args, **kwargs: P.kwargs) -> None: ...
    @overload
    def assertWarns[W: Warning](self, expected_warning: type[W], *,
                                msg: str | None = None) -> _AssertWarnsContext[W]: ...
    def assertWarns[W: Warning, **P](self, expected_warning: type[W],
                                     callable_obj: Callable[P, object] | None = None,
                                     *args: P.args, msg: str | None = None,
                                     **kwargs: P.kwargs) -> _AssertWarnsContext[W] | None:
        """Fail unless a warning of class warnClass is triggered
           by the callable when invoked with specified positional and
           keyword arguments.  If a different type of warning is
           triggered, it will not be handled: depending on the other
           warning filtering rules in effect, it might be silenced, printed
           out, or raised as an exception.

           If called with the callable and arguments omitted, will return a
           context object used like this::

                with self.assertWarns(SomeWarning):
                    do_something()

           An optional keyword argument 'msg' can be provided when assertWarns
           is used as a context object.

           The context manager keeps a reference to the first matching
           warning as the 'warning' attribute; similarly, the 'filename'
           and 'lineno' attributes give you information about the line
           of Python code from which the warning was triggered.
           This allows you to inspect the warning after the assertion::

               with self.assertWarns(SomeWarning) as cm:
                   do_something()
               the_warning = cm.warning
               self.assertEqual(the_warning.some_attribute, 147)
        """
        context = _AssertWarnsContext[W](expected_warning, self, None, msg)
        if callable_obj is None:
            return context
        context.obj_name = callable_obj.__name__
        with context:
            callable_obj(*args, **kwargs)
        return None

    @overload
    def assertWarnsRegex[**P](self, expected_warning: type[Warning],
                              expected_regex: str | re.Pattern,
                              callable: Callable[P, object], /,
                              *args: P.args, **kwargs: P.kwargs) -> None: ...
    @overload
    def assertWarnsRegex[W: Warning](self, expected_warning: type[W],
                                     expected_regex: str | re.Pattern, *,
                                     msg: str | None = None) -> _AssertWarnsContext[W]: ...
    def assertWarnsRegex[W: Warning, **P](self, expected_warning: type[W],
                                          expected_regex: str | re.Pattern,
                                          callable_obj: Callable[P, object] | None = None,
                                          *args: P.args, msg: str | None = None,
                                          **kwargs: P.kwargs) -> _AssertWarnsContext[W] | None:
        """Asserts that the message in a triggered warning matches a regexp.
        Basic functioning is similar to assertWarns() with the addition
        that only warnings whose messages also match the regular expression
        are considered successful matches.

        Args:
            expected_warning: Warning class expected to be triggered.
            expected_regex: Regex (re.Pattern object or string) expected
                    to be found in error message.
            args: Function to be called and extra positional args.
            kwargs: Extra kwargs.
            msg: Optional message used in case of failure. Can only be used
                    when assertWarnsRegex is used as a context manager.
        """
        context = _AssertWarnsContext[W](expected_warning, self, expected_regex, msg)
        if callable_obj is None:
            return context
        context.obj_name = callable_obj.__name__
        with context:
            callable_obj(*args, **kwargs)
        return None

    def assertLogs(self, logger: str | logging.Logger | None = None,
                   level: int | str | None = None) -> _AssertLogsContext:
        """Fail unless a log message of level *level* or higher is emitted
        on *logger_name* or its children.  If omitted, *level* defaults to
        INFO and *logger* defaults to the root logger.

        This method must be used as a context manager, and will yield
        a recording object with two attributes: `output` and `records`.
        At the end of the context manager, the `output` attribute will
        be a list of the matching formatted log messages and the
        `records` attribute will be a list of the corresponding LogRecord
        objects.
        """
        return _AssertLogsContext(self, logger, level)

    def assertNoLogs(self, logger: str | logging.Logger | None = None,
                     level: int | str | None = None) -> _AssertNoLogsContext:
        """ Fail unless no log messages of level *level* or higher are emitted
        on *logger_name* or its children.

        This method must be used as a context manager.
        """
        return _AssertNoLogsContext(self, logger, level)

    def assertRegex(self, text: str, expected_regex: str | re.Pattern,
                    msg: str | None = None) -> None:
        """Fail the test unless the text matches the regular expression."""
        if isinstance(expected_regex, str):
            assert expected_regex, "expected_regex must not be empty."
        regex = re.compile(expected_regex)
        if regex.search(text) is None:
            standardMsg = "Regex didn't match: %r not found in %r" % (
                regex.pattern, text)
            # _formatMessage ensures the longMessage option is respected
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)

    def assertNotRegex(self, text: str, unexpected_regex: str | re.Pattern,
                       msg: str | None = None) -> None:
        """Fail the test if the text matches the regular expression."""
        regex = re.compile(unexpected_regex)
        match = regex.search(text)
        if match is not None:
            standardMsg = 'Regex matched: %r matches %r in %r' % (
                text[match.start() : match.end()],
                regex.pattern,
                text)
            # _formatMessage ensures the longMessage option is respected
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)

    def _baseAssertEqual[T](self, first: T, second: T, msg: str | None = None) -> None:
        """The default assertEqual implementation, not type specific."""
        if not first == second:
            standardMsg = '%s != %s' % _common_shorten_repr(first, second)
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)

    def assertEqual[T](self, first: T, second: T, msg: str | None = None) -> None:
        """Fail if the two objects are unequal as determined by the '=='
           operator.
        """
        if isinstance(first, str) and isinstance(second, str):
            self.assertMultiLineEqual(first, second, msg)
            return
        if isinstance(first, list) and isinstance(second, list):
            self.assertListEqual(first, second, msg)
            return
        if isinstance(first, tuple) and isinstance(second, tuple):
            self.assertTupleEqual(first, second, msg)
            return
        if isinstance(first, dict) and isinstance(second, dict):
            self.assertDictEqual(first, second, msg)
            return
        if isinstance(first, set) and isinstance(second, set):
            self.assertSetEqual(first, second, msg)
            return
        if isinstance(first, frozenset) and isinstance(second, frozenset):
            self.assertSetEqual(first, second, msg)
            return
        self._baseAssertEqual(first, second, msg)

    def assertNotEqual[T](self, first: T, second: T, msg: str | None = None) -> None:
        """Fail if the two objects are equal as determined by the '!='
           operator.
        """
        if not first != second:
            msg = self._formatMessage(msg, '%s == %s' % (safe_repr(first),
                                                          safe_repr(second)))
            raise self.failureException(msg)

    def assertAlmostEqual(self, first: float, second: float, places: int | None = None,
                          msg: str | None = None, delta: float | None = None) -> None:
        """Fail if the two objects are unequal as determined by their
           difference rounded to the given number of decimal places
           (default 7) and comparing to zero, or by comparing that the
           difference between the two objects is more than the given
           delta.

           Note that decimal places (from zero) are usually not the same
           as significant digits (measured from the most significant digit).

           If the two objects compare equal then they will automatically
           compare almost equal.
        """
        if first == second:
            # shortcut
            return
        if delta is not None and places is not None:
            raise TypeError("specify delta or places not both")

        diff = abs(first - second)
        if delta is not None:
            if diff <= delta:
                return

            standardMsg = '%s != %s within %s delta (%s difference)' % (
                safe_repr(first),
                safe_repr(second),
                safe_repr(delta),
                safe_repr(diff))
        else:
            if places is None:
                places = 7

            if round(diff, places) == 0:
                return

            standardMsg = '%s != %s within %r places (%s difference)' % (
                safe_repr(first),
                safe_repr(second),
                places,
                safe_repr(diff))
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def assertSequenceEqual[S](self, seq1: S, seq2: S, msg: str | None = None,
                               seq_type: type | None = None) -> None:
        """An equality assertion for ordered sequences (like lists and tuples).

        For the purposes of this function, a valid ordered sequence type is one
        which can be indexed, has a length, and has an equality operator.

        Args:
            seq1: The first sequence to compare.
            seq2: The second sequence to compare.
            seq_type: The expected datatype of the sequences, or None if no
                    datatype should be enforced.
            msg: Optional message to use on failure instead of a list of
                    differences.
        """
        if seq_type is not None:
            seq_type_name = seq_type.__name__
        else:
            seq_type_name = "sequence"

        len1 = len(seq1)
        len2 = len(seq2)

        if seq1 == seq2:
            return

        differing = '%ss differ: %s != %s\n' % (
                (seq_type_name.capitalize(),) +
                _common_shorten_repr(seq1, seq2))

        for i in range(min(len1, len2)):
            item1 = seq1[i]
            item2 = seq2[i]

            if item1 != item2:
                differing += ('\nFirst differing element %d:\n%s\n%s\n' %
                             ((i,) + _common_shorten_repr(item1, item2)))
                break
        else:
            if (len1 == len2 and seq_type is None and
                type(seq1) != type(seq2)):
                # The sequences are the same, but have differing types.
                return

        if len1 > len2:
            differing += ('\nFirst %s contains %d additional '
                         'elements.\n' % (seq_type_name, len1 - len2))
            differing += ('First extra element %d:\n%s\n' %
                          (len2, safe_repr(seq1[len2])))
        elif len1 < len2:
            differing += ('\nSecond %s contains %d additional '
                         'elements.\n' % (seq_type_name, len2 - len1))
            differing += ('First extra element %d:\n%s\n' %
                          (len1, safe_repr(seq2[len1])))
        standardMsg = differing
        diffMsg = '\n' + '\n'.join(
            difflib.ndiff(pprint.pformat(seq1).splitlines(),
                          pprint.pformat(seq2).splitlines()))

        standardMsg = self._truncateMessage(standardMsg, diffMsg)
        msg = self._formatMessage(msg, standardMsg)
        self.fail(msg)

    def _truncateMessage(self, message: str, diff: str) -> str:
        max_diff = self.maxDiff
        if max_diff is None or len(diff) <= max_diff:
            return message + diff
        return message + (DIFF_OMITTED % len(diff))

    def assertListEqual[T](self, list1: list[T], list2: list[T], msg: str | None = None) -> None:
        """A list-specific equality assertion.

        Args:
            list1: The first list to compare.
            list2: The second list to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.

        """
        self.assertSequenceEqual(list1, list2, msg, seq_type=list)

    def assertTupleEqual[S](self, tuple1: S, tuple2: S, msg: str | None = None) -> None:
        """A tuple-specific equality assertion.

        Args:
            tuple1: The first tuple to compare.
            tuple2: The second tuple to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.
        """
        self.assertSequenceEqual(tuple1, tuple2, msg, seq_type=tuple)

    def assertSetEqual[S](self, set1: S, set2: S, msg: str | None = None) -> None:
        """A set-specific equality assertion.

        Args:
            set1: The first set to compare.
            set2: The second set to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.

        assertSetEqual uses ducktyping to support different types of sets, and
        is optimized for sets specifically (parameters must support a
        difference method).
        """
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

        standardMsg = '\n'.join(lines)
        self.fail(self._formatMessage(msg, standardMsg))

    def assertIn[T](self, member: T, container: list[T], msg: str | None = None) -> None:
        """Just like self.assertTrue(a in b), but with a nicer default message."""
        if member not in container:
            standardMsg = '%s not found in %s' % (safe_repr(member),
                                                  safe_repr(container))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertNotIn[T](self, member: T, container: list[T], msg: str | None = None) -> None:
        """Just like self.assertTrue(a not in b), but with a nicer default message."""
        if member in container:
            standardMsg = '%s unexpectedly found in %s' % (safe_repr(member),
                                                        safe_repr(container))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIs(self, expr1: object, expr2: object, msg: str | None = None) -> None:
        """Just like self.assertTrue(a is b), but with a nicer default message."""
        if expr1 is not expr2:
            standardMsg = '%s is not %s' % (safe_repr(expr1),
                                             safe_repr(expr2))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIsNot(self, expr1: object, expr2: object, msg: str | None = None) -> None:
        """Just like self.assertTrue(a is not b), but with a nicer default message."""
        if expr1 is expr2:
            standardMsg = 'unexpectedly identical: %s' % (safe_repr(expr1),)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertDictEqual[K, V](self, d1: dict[K, V], d2: dict[K, V], msg: str | None = None) -> None:
        if d1 != d2:
            standardMsg = '%s != %s' % _common_shorten_repr(d1, d2)
            diff = ('\n' + '\n'.join(difflib.ndiff(
                           pprint.pformat(d1).splitlines(),
                           pprint.pformat(d2).splitlines())))
            standardMsg = self._truncateMessage(standardMsg, diff)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertMultiLineEqual(self, first: str, second: str, msg: str | None = None) -> None:
        """Assert that two multi-line strings are equal."""
        if first != second:
            # Don't use difflib if the strings are too long
            if (len(first) > self._diffThreshold or
                len(second) > self._diffThreshold):
                self._baseAssertEqual(first, second, msg)

            # Append \n to both strings if either is missing the \n.
            # This allows the final ndiff to show the \n difference. The
            # exception here is if the string is empty, in which case no
            # \n should be added
            first_presplit = first
            second_presplit = second
            if first and second:
                if first[-1] != '\n' or second[-1] != '\n':
                    first_presplit += '\n'
                    second_presplit += '\n'
            elif second and second[-1] != '\n':
                second_presplit += '\n'
            elif first and first[-1] != '\n':
                first_presplit += '\n'

            firstlines = first_presplit.splitlines(keepends=True)
            secondlines = second_presplit.splitlines(keepends=True)

            # Generate the message and diff, then raise the exception
            standardMsg = '%s != %s' % _common_shorten_repr(first, second)
            diff = '\n' + ''.join(difflib.ndiff(firstlines, secondlines))
            standardMsg = self._truncateMessage(standardMsg, diff)
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

    def assertIsNone(self, obj: object, msg: str | None = None) -> None:
        """Same as self.assertTrue(obj is None), with a nicer default message."""
        if obj is not None:
            standardMsg = '%s is not None' % (safe_repr(obj),)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIsNotNone(self, obj: object, msg: str | None = None) -> None:
        """Included for symmetry with assertIsNone."""
        if obj is None:
            standardMsg = 'unexpectedly None'
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIsInstance(self, obj: object, cls: type, msg: str | None = None) -> None:
        """Same as self.assertTrue(isinstance(obj, cls)), with a nicer
        default message."""
        if not isinstance(obj, cls):
            standardMsg = f'{safe_repr(obj)} is not an instance of {cls!r}'
            self.fail(self._formatMessage(msg, standardMsg))



class FunctionTestCase(TestCase):
    """A test case that wraps a test function.

    This is useful for slipping pre-existing test functions into the
    unittest framework. Optionally, set-up and tidy-up functions can be
    supplied. As with TestCase, the tidy-up ('tearDown') function will
    always be called if the set-up ('setUp') function ran successfully.
    """

    def __init__(self, testFunc: Callable[[], object], setUp: Callable[[], object] | None = None,
                 tearDown: Callable[[], object] | None = None,
                 description: str | None = None) -> None:
        super(FunctionTestCase, self).__init__()
        self._setUpFunc = setUp
        self._tearDownFunc = tearDown
        self._testFunc = testFunc
        self._description = description

    def setUp(self) -> None:
        if self._setUpFunc is not None:
            self._setUpFunc()

    def tearDown(self) -> None:
        if self._tearDownFunc is not None:
            self._tearDownFunc()

    def runTest(self) -> None:
        self._testFunc()

    def id(self) -> str:
        return self._testFunc.__name__

    def __str__(self) -> str:
        return "%s (%s)" % (strclass(self.__class__),
                            self._testFunc.__name__)

    def __repr__(self) -> str:
        return "<%s tec=%s>" % (strclass(self.__class__),
                                     self._testFunc.__name__)

    def shortDescription(self) -> str | None:
        if self._description is not None:
            return self._description
        doc = self._testFunc.__doc__
        return doc and doc.split("\n")[0].strip() or None


class _SubTest(TestCase):

    def __init__(self, test_case: TestCase, message: object, params: dict[str, object]) -> None:
        super().__init__()
        self._message = message
        self.test_case = test_case
        self.params = params
        self.failureException = test_case.failureException

    def runTest(self) -> None:
        raise NotImplementedError("subtests cannot be run directly")

    def _subDescription(self) -> str:
        parts: list[str] = []
        if self._message is not _subtest_msg_sentinel:
            parts.append("[{}]".format(self._message))
        if self.params:
            params_desc = ', '.join(
                "{}={!r}".format(k, v)
                for (k, v) in self.params.items())
            parts.append("({})".format(params_desc))
        return " ".join(parts) or '(<subtest>)'

    def id(self) -> str:
        return "{} {}".format(self.test_case.id(), self._subDescription())

    def shortDescription(self) -> str | None:
        """Returns a one-line description of the subtest, or None if no
        description has been provided.
        """
        return self.test_case.shortDescription()

    def __str__(self) -> str:
        return "{} {}".format(self.test_case, self._subDescription())
