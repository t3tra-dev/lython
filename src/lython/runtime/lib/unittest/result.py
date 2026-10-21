"""Test result object"""

import sys
import traceback
import types

from . import case
from . import util

__unittest = True

_SysExcInfoType = tuple[type[BaseException], BaseException, types.TracebackType]


class TestResult(object):
    """Holder for test result information.

    Test results are automatically managed by the TestCase and TestSuite
    classes, and do not need to be explicitly manipulated by writers of tests.

    Each instance holds the total number of tests run, and collections of
    failures and errors that occurred among those test runs. The collections
    contain tuples of (testcase, exceptioninfo), where exceptioninfo is the
    formatted traceback of the error that occurred.
    """
    _previousTestClass: type[case.TestCase] | None = None
    _testRunEntered = False
    _moduleSetUpFailed = False
    def __init__(self, stream: object = None, descriptions: bool | None = None,
                 verbosity: int | None = None) -> None:
        self.failfast = False
        self.failures: list[tuple[case.TestCase, str]] = []
        self.errors: list[tuple[case.TestCase, str]] = []
        self.testsRun = 0
        self.skipped: list[tuple[case.TestCase, str]] = []
        self.expectedFailures: list[tuple[case.TestCase, str]] = []
        self.unexpectedSuccesses: list[case.TestCase] = []
        self.collectedDurations: list[tuple[str, float]] = []
        self.shouldStop = False
        self.buffer = False
        self.tb_locals = False
        self._classSetupFailed: dict[type[case.TestCase], bool] = {}

    def printErrors(self) -> None:
        "Called by TestRunner after test run"

    def startTest(self, test: case.TestCase) -> None:
        "Called when the given test is about to be run"
        self.testsRun += 1

    def startTestRun(self) -> None:
        """Called once before any tests are executed.

        See startTest for a method called before each test.
        """

    def stopTest(self, test: case.TestCase) -> None:
        """Called when the given test has been run"""

    def stopTestRun(self) -> None:
        """Called once after all tests are executed.

        See stopTest for a method called after each test.
        """

    def addError(self, test: case.TestCase, err: _SysExcInfoType) -> None:
        """Called when an error has occurred. 'err' is a tuple of values as
        returned by sys.exc_info().
        """
        self.errors.append((test, self._exc_info_to_string(err, test)))
        if self.failfast:
            self.stop()

    def addFailure(self, test: case.TestCase, err: _SysExcInfoType) -> None:
        """Called when an error has occurred. 'err' is a tuple of values as
        returned by sys.exc_info()."""
        self.failures.append((test, self._exc_info_to_string(err, test)))
        if self.failfast:
            self.stop()

    def addSubTest(self, test: case.TestCase, subtest: case.TestCase,
                   err: _SysExcInfoType | None) -> None:
        """Called at the end of a subtest.
        'err' is None if the subtest ended successfully, otherwise it's a
        tuple of values as returned by sys.exc_info().
        """
        # By default, we don't do anything with successful subtests, but
        # more sophisticated test results might want to record them.
        if err is not None:
            if self.failfast:
                self.stop()
            if issubclass(err[0], test.failureException):
                errors = self.failures
            else:
                errors = self.errors
            errors.append((subtest, self._exc_info_to_string(err, test)))

    def addSuccess(self, test: case.TestCase) -> None:
        "Called when a test has completed successfully"
        pass

    def addSkip(self, test: case.TestCase, reason: str) -> None:
        """Called when a test is skipped."""
        self.skipped.append((test, reason))

    def addExpectedFailure(self, test: case.TestCase, err: _SysExcInfoType) -> None:
        """Called when an expected failure/error occurred."""
        self.expectedFailures.append(
            (test, self._exc_info_to_string(err, test)))

    def addUnexpectedSuccess(self, test: case.TestCase) -> None:
        """Called when a test was expected to fail, but succeed."""
        self.unexpectedSuccesses.append(test)
        if self.failfast:
            self.stop()

    def addDuration(self, test: case.TestCase, elapsed: float) -> None:
        """Called when a test finished to run, regardless of its outcome.
        *test* is the test case corresponding to the test method.
        *elapsed* is the time represented in seconds, and it includes the
        execution of cleanup functions.
        """
        # support for a TextTestRunner using an old TestResult class
        if hasattr(self, "collectedDurations"):
            # Pass test repr and not the test object itself to avoid resources leak
            self.collectedDurations.append((str(test), elapsed))

    def wasSuccessful(self) -> bool:
        """Tells whether or not this result was a success."""
        # The hasattr check is for test_result's OldResult test.  That
        # way this method works on objects that lack the attribute.
        # (where would such result instances come from? old stored pickles?)
        return ((len(self.failures) == len(self.errors) == 0) and
                (not hasattr(self, 'unexpectedSuccesses') or
                 len(self.unexpectedSuccesses) == 0))

    def stop(self) -> None:
        """Indicates that the tests should be aborted."""
        self.shouldStop = True

    def _exc_info_to_string(self, err: _SysExcInfoType, test: case.TestCase) -> str:
        """Converts a sys.exc_info()-style tuple of values into a string."""
        exctype, value, tb = err
        entries = traceback.format_tb(tb)
        # Skip test runner traceback levels, then (for a failure) the assert*
        # levels below the test's own frame.
        start = 0
        while start < len(entries) and _is_relevant_tb_level(entries[start]):
            start += 1
        end = len(entries)
        if exctype is test.failureException:
            cut = start
            while cut < end and not _is_relevant_tb_level(entries[cut]):
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

    def __repr__(self) -> str:
        return ("<%s run=%i errors=%i failures=%i>" %
               (util.strclass(self.__class__), self.testsRun, len(self.errors),
                len(self.failures)))


def _is_relevant_tb_level(entry: str) -> bool:
    first = entry.split("\n")[0]
    return '<stdlib>/unittest/' in first
