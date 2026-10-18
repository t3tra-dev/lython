# WHAT: a module the next case imports -- a TestCase that runs the method its
# instance names, found by name with `hasattr` and called with `getattr`, and
# a loader that lists a class's test methods with `dir`. Run alone it prints
# nothing.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: it is the imported half of
# a_test_case_found_and_run_by_its_name.py, which needs it on disk; running it
# alone keeps it a program of its own.
class TestCase:
    def __init__(self, methodName: str = "runTest") -> None:
        self._testMethodName = methodName
        self.log: list[str] = []

    def setUp(self) -> None:
        self.log.append("setUp")

    def tearDown(self) -> None:
        self.log.append("tearDown")

    def run(self) -> str:
        if not hasattr(self, self._testMethodName):
            return "no such test method: " + self._testMethodName
        self.setUp()
        try:
            getattr(self, self._testMethodName)()
            outcome = "ok"
        except AssertionError as e:
            outcome = "FAIL: " + str(e)
        except Exception as e:
            outcome = "ERROR: " + type(e).__name__ + ": " + str(e)
        finally:
            self.tearDown()
        return outcome


def test_names(cls: type[TestCase]) -> list[str]:
    return [name for name in dir(cls) if name.startswith("test")]


def load(cls: type[TestCase]) -> list[TestCase]:
    return [cls(name) for name in test_names(cls)]
