"""TestSuite"""

import sys
from collections.abc import Iterable, Iterator

from . import case
from . import result
from . import util

__unittest = True

_SysExcInfoType = case._SysExcInfoType


class BaseTestSuite(object):
    """A simple test suite that doesn't provide class or module shared fixtures.
    """
    _cleanup = True

    def __init__(self, tests: Iterable[case.TestCase | BaseTestSuite] = ()) -> None:
        self._tests: list[case.TestCase | BaseTestSuite] = []
        self._removed_tests = 0
        self.addTests(tests)

    def __repr__(self) -> str:
        return "<%s tests=%s>" % (util.strclass(self.__class__), list(self))

    def __iter__(self) -> Iterator[case.TestCase | BaseTestSuite]:
        return iter(self._tests)

    def countTestCases(self) -> int:
        cases = self._removed_tests
        for test in self:
            if test:
                cases += test.countTestCases()
        return cases

    def addTest(self, test: case.TestCase | BaseTestSuite) -> None:
        self._tests.append(test)

    def addTests(self, tests: Iterable[case.TestCase | BaseTestSuite]) -> None:
        if isinstance(tests, str):
            raise TypeError("tests must be an iterable of tests, not a string")
        for test in tests:
            self.addTest(test)

    def run(self, result: result.TestResult) -> result.TestResult:
        for index, test in enumerate(self):
            if result.shouldStop:
                break
            test(result)
        return result

    def __call__(self, result: result.TestResult) -> result.TestResult:
        return self.run(result)


class TestSuite(BaseTestSuite):
    """A test suite is a composite test consisting of a number of TestCases.

    For use, create an instance of TestSuite, then add test case instances.
    When all tests have been added, the suite can be passed to a test
    runner, such as TextTestRunner. It will run the individual test cases
    in the order in which they were added, aggregating the results. When
    subclassing, do not forget to call the base class constructor.
    """

    def run(self, result: result.TestResult, debug: bool = False) -> result.TestResult:
        topLevel = False
        if getattr(result, '_testRunEntered', False) is False:
            result._testRunEntered = topLevel = True

        for index, test in enumerate(self):
            if result.shouldStop:
                break

            if isinstance(test, case.TestCase):
                self._tearDownPreviousClass(test, result)
                self._handleClassSetUp(test, result)
                result._previousTestClass = test.__class__

                if result._classSetupFailed.get(test.__class__, False):
                    continue

            test(result)

        if topLevel:
            self._tearDownPreviousClass(None, result)
            result._testRunEntered = False
        return result

    ################################

    def _handleClassSetUp(self, test: case.TestCase, result: result.TestResult) -> None:
        previousClass = getattr(result, '_previousTestClass', None)
        currentClass = test.__class__
        if currentClass == previousClass:
            return

        result._classSetupFailed[currentClass] = False
        try:
            currentClass.setUpClass()
        except Exception as e:
            result._classSetupFailed[currentClass] = True
            className = util.strclass(currentClass)
            self._createClassOrModuleLevelException(result, e,
                                                    'setUpClass',
                                                    className)

    def _createClassOrModuleLevelException(self, result: result.TestResult, exc: Exception,
                                           method_name: str, parent: str,
                                           info: _SysExcInfoType | None = None) -> None:
        errorName = f'{method_name} ({parent})'
        self._addClassOrModuleLevelException(result, exc, errorName, info)

    def _addClassOrModuleLevelException(self, result: result.TestResult, exception: Exception,
                                        errorName: str,
                                        info: _SysExcInfoType | None = None) -> None:
        error = _ErrorHolder(errorName)
        if isinstance(exception, case.SkipTest):
            result.addSkip(error, str(exception))
        else:
            if not info:
                exc_type, exc_value, tb = sys.exc_info()
                if exc_type is not None and exc_value is not None and tb is not None:
                    result.addError(error, (exc_type, exc_value, tb))
            else:
                result.addError(error, info)

    def _tearDownPreviousClass(self, test: case.TestCase | None, result: result.TestResult) -> None:
        previousClass = getattr(result, '_previousTestClass', None)
        if previousClass is None:
            return
        if test is not None and test.__class__ is previousClass:
            return
        if result._classSetupFailed.get(previousClass, False):
            return

        try:
            previousClass.tearDownClass()
        except Exception as e:
            className = util.strclass(previousClass)
            self._createClassOrModuleLevelException(result, e,
                                                    'tearDownClass',
                                                    className)


class _ErrorHolder(case.TestCase):
    """
    Placeholder for a TestCase inside a result. As far as a TestResult
    is concerned, this looks exactly like a unit test. Used to insert
    arbitrary errors into a test suite run.
    """
    # Inspired by the ErrorHolder from Twisted:
    # http://twistedmatrix.com/trac/browser/trunk/twisted/trial/runner.py

    def __init__(self, description: str) -> None:
        super().__init__()
        self.description = description

    def runTest(self) -> None:
        pass

    def id(self) -> str:
        return self.description

    def shortDescription(self) -> str | None:
        return None

    def __repr__(self) -> str:
        return "<ErrorHolder description=%r>" % (self.description,)

    def __str__(self) -> str:
        return self.id()

    def countTestCases(self) -> int:
        return 0
