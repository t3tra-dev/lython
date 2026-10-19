# WHAT: A TestCase with a classmethod, and a runner that calls a test method
#   by a computed name. Run alone it prints nothing.
# KIND: helper module of ut_bug_computed_getattr_classmethod_arm_in_importer
class TestCase:
    def __init__(self, methodName: str = "runTest") -> None:
        self._testMethodName = methodName

    @classmethod
    def setUpClass(cls) -> None:
        pass


def run_case(test: TestCase) -> None:
    getattr(test, test._testMethodName)()
