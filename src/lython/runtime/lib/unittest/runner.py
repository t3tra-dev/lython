"""Running tests"""

import io
import sys
import time

from . import case
from . import result
from . import suite
from .case import _SubTest

__unittest = True

_SysExcInfoType = case._SysExcInfoType


class _WritelnDecorator(object):
    """Used to decorate file-like objects with a handy 'writeln' method"""
    def __init__(self, stream: io.TextIOWrapper | io.StringIO) -> None:
        self.stream = stream

    def write(self, arg: str) -> None:
        self.stream.write(arg)

    def flush(self) -> None:
        self.stream.flush()

    def writeln(self, arg: str | None = None) -> None:
        if arg:
            self.write(arg)
        self.write('\n')  # text-mode streams translate to \r\n if needed


class TextTestResult(result.TestResult):
    """A test result class that can print formatted text results to a stream.

    Used by TextTestRunner.
    """
    separator1 = '=' * 70
    separator2 = '-' * 70

    def __init__(self, stream: _WritelnDecorator, descriptions: bool, verbosity: int, *,
                 durations: int | None = None) -> None:
        """Construct a TextTestResult. Subclasses should accept **kwargs
        to ensure compatibility as the interface changes."""
        super(TextTestResult, self).__init__(stream, descriptions, verbosity)
        self.stream = stream
        self.showAll = verbosity > 1
        self.dots = verbosity == 1
        self.descriptions = descriptions
        self._newline = True
        self.durations = durations

    def getDescription(self, test: case.TestCase) -> str:
        doc_first_line = test.shortDescription()
        if self.descriptions and doc_first_line:
            return '\n'.join((str(test), doc_first_line))
        else:
            return str(test)

    def startTest(self, test: case.TestCase) -> None:
        super(TextTestResult, self).startTest(test)
        if self.showAll:
            self.stream.write(self.getDescription(test))
            self.stream.write(" ... ")
            self.stream.flush()
            self._newline = False

    def _write_status(self, test: case.TestCase, status: str) -> None:
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

    def addSubTest(self, test: case.TestCase, subtest: case.TestCase,
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
        super(TextTestResult, self).addSubTest(test, subtest, err)

    def addSuccess(self, test: case.TestCase) -> None:
        super(TextTestResult, self).addSuccess(test)
        if self.showAll:
            self._write_status(test, "ok")
        elif self.dots:
            self.stream.write(".")
            self.stream.flush()

    def addError(self, test: case.TestCase, err: _SysExcInfoType) -> None:
        super(TextTestResult, self).addError(test, err)
        if self.showAll:
            self._write_status(test, "ERROR")
        elif self.dots:
            self.stream.write("E")
            self.stream.flush()

    def addFailure(self, test: case.TestCase, err: _SysExcInfoType) -> None:
        super(TextTestResult, self).addFailure(test, err)
        if self.showAll:
            self._write_status(test, "FAIL")
        elif self.dots:
            self.stream.write("F")
            self.stream.flush()

    def addSkip(self, test: case.TestCase, reason: str) -> None:
        super(TextTestResult, self).addSkip(test, reason)
        if self.showAll:
            self._write_status(test, "skipped {0!r}".format(reason))
        elif self.dots:
            self.stream.write("s")
            self.stream.flush()

    def addExpectedFailure(self, test: case.TestCase, err: _SysExcInfoType) -> None:
        super(TextTestResult, self).addExpectedFailure(test, err)
        if self.showAll:
            self.stream.writeln("expected failure")
            self.stream.flush()
        elif self.dots:
            self.stream.write("x")
            self.stream.flush()

    def addUnexpectedSuccess(self, test: case.TestCase) -> None:
        super(TextTestResult, self).addUnexpectedSuccess(test)
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
        self.printErrorList('ERROR', self.errors)
        self.printErrorList('FAIL', self.failures)
        unexpectedSuccesses = getattr(self, 'unexpectedSuccesses', ())
        if unexpectedSuccesses:
            self.stream.writeln(self.separator1)
            for test in unexpectedSuccesses:
                self.stream.writeln(
                    f"UNEXPECTED SUCCESS: "
                    f"{self.getDescription(test)}"
                )
            self.stream.flush()

    def printErrorList(self, flavour: str, errors: list[tuple[case.TestCase, str]]) -> None:
        for test, err in errors:
            self.stream.writeln(self.separator1)
            self.stream.writeln(
                f"{flavour}: {self.getDescription(test)}"
            )
            self.stream.writeln(self.separator2)
            self.stream.writeln("%s" % err)
            self.stream.flush()


class TextTestRunner(object):
    """A test runner class that displays results in textual form.

    It prints out the names of tests as they are run, errors as they
    occur, and a summary of the results at the end of the test run.
    """
    resultclass = TextTestResult

    def __init__(self, stream: io.TextIOWrapper | io.StringIO | None = None,
                 descriptions: bool = True, verbosity: int = 1,
                 failfast: bool = False, buffer: bool = False,
                 resultclass: type[TextTestResult] | None = None,
                 warnings: str | None = None,
                 *, tb_locals: bool = False, durations: int | None = None) -> None:
        """Construct a TextTestRunner.

        Subclasses should accept **kwargs to ensure compatibility as the
        interface changes.
        """
        if stream is None:
            stream = sys.stderr
        self.stream = _WritelnDecorator(stream)
        self.descriptions = descriptions
        self.verbosity = verbosity
        self.failfast = failfast
        self.buffer = buffer
        self.tb_locals = tb_locals
        self.durations = durations
        self.warnings = warnings
        if resultclass is not None:
            self.resultclass = resultclass

    def _makeResult(self) -> TextTestResult:
        return self.resultclass(self.stream, self.descriptions,
                                self.verbosity, durations=self.durations)

    def _printDurations(self, result: TextTestResult) -> None:
        if not result.collectedDurations:
            return
        ls = sorted(result.collectedDurations, key=lambda x: x[1],
                    reverse=True)
        durations = self.durations
        if durations is not None and durations > 0:
            ls = ls[:durations]
        self.stream.writeln("Slowest test durations")
        self.stream.writeln(result.separator2)
        hidden = False
        for test, elapsed in ls:
            if self.verbosity < 2 and elapsed < 0.001:
                hidden = True
                continue
            self.stream.writeln("%-10s %s" % ("%.3fs" % elapsed, test))
        if hidden:
            self.stream.writeln("\n(durations < 0.001s were hidden; "
                                "use -v to show these durations)")
        else:
            self.stream.writeln("")

    def run(self, test: suite.BaseTestSuite | case.TestCase) -> TextTestResult:
        "Run the given test case or test suite."
        result = self._makeResult()
        result.failfast = self.failfast
        result.buffer = self.buffer
        result.tb_locals = self.tb_locals
        start_time = time.perf_counter()
        result.startTestRun()
        try:
            test(result)
        finally:
            result.stopTestRun()
        stop_time = time.perf_counter()
        time_taken = stop_time - start_time
        result.printErrors()
        if self.durations is not None:
            self._printDurations(result)

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
                infos.append(f"failures={failed}")
            if errored != 0:
                infos.append(f"errors={errored}")
        elif run == 0 and skipped == 0:
            self.stream.write("NO TESTS RAN")
        else:
            self.stream.write("OK")
        if skipped != 0:
            infos.append(f"skipped={skipped}")
        if expected_fails != 0:
            infos.append(f"expected failures={expected_fails}")
        if unexpected_successes != 0:
            infos.append(f"unexpected successes={unexpected_successes}")
        if infos:
            self.stream.writeln(" (%s)" % (", ".join(infos),))
        else:
            self.stream.write("\n")
        self.stream.flush()
        return result
