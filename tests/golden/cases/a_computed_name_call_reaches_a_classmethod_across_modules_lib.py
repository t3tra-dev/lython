class TestCase:
    def __init__(self, methodName: str = "runTest") -> None:
        self._testMethodName = methodName

    @classmethod
    def setUpClass(cls) -> None:
        print("setUpClass of", cls.__name__)

    @classmethod
    def kind(cls) -> str:
        return cls.__name__


class Special(TestCase):
    pass


def run_case(test: TestCase) -> None:
    getattr(test, test._testMethodName)()
