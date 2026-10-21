"""Loading unittests."""

from . import case, suite, util

__unittest = True


class TestLoader(object):
    """
    This class is responsible for loading tests according to various criteria
    and returning them wrapped in a TestSuite
    """
    testMethodPrefix = 'test'
    testNamePatterns: list[str] | None = None

    def __init__(self) -> None:
        super(TestLoader, self).__init__()
        self.errors: list[str] = []

    def loadTestsFromTestCase(self, testCaseClass: type[case.TestCase]) -> suite.TestSuite:
        """Return a suite of all test cases contained in testCaseClass"""
        if testCaseClass is case.TestCase or testCaseClass is case.FunctionTestCase:
            # We don't load any tests from base types that should not be loaded
            testCaseNames: list[str] = []
        else:
            testCaseNames = self.getTestCaseNames(testCaseClass)
            if not testCaseNames and hasattr(testCaseClass, 'runTest'):
                testCaseNames = ['runTest']
        loaded_suite = suite.TestSuite(map(testCaseClass, testCaseNames))
        return loaded_suite

    def loadTestsFromModule(self, module: str, *, pattern: str | None = None) -> suite.TestSuite:
        """Return a suite of all test cases contained in the given module"""
        tests: list[case.TestCase | suite.BaseTestSuite] = []
        found = [cls for cls in _all_subclasses(case.TestCase) if cls.__module__ == module]
        for obj in sorted(found, key=_class_name):
            tests.append(self.loadTestsFromTestCase(obj))
        return suite.TestSuite(tests)

    def getTestCaseNames(self, testCaseClass: type[case.TestCase]) -> list[str]:
        """Return a sorted sequence of method names found within testCaseClass
        """
        def shouldIncludeMethod(attrname: str) -> bool:
            if not attrname.startswith(self.testMethodPrefix):
                return False
            fullName = f'%s.%s.%s' % (
                testCaseClass.__module__, testCaseClass.__qualname__, attrname
            )
            patterns = self.testNamePatterns
            return patterns is None or \
                any(pattern in fullName for pattern in patterns)
        testFnNames = list(filter(shouldIncludeMethod, dir(testCaseClass)))
        testFnNames.sort()
        return testFnNames


defaultTestLoader = TestLoader()


def _all_subclasses(cls: type[case.TestCase]) -> list[type[case.TestCase]]:
    found: list[type[case.TestCase]] = []
    for sub in cls.__subclasses__():
        found.append(sub)
        found.extend(_all_subclasses(sub))
    return found


def _class_name(cls: type[case.TestCase]) -> str:
    return cls.__name__
