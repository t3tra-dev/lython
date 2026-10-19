# WHAT: `cls(...)` through a `type[TestCase]` value, where the program's
# subclasses declare `__init__`s that take other arguments -- more of them,
# keyword-only ones, a differently spelled one. A class that can take the
# call is built; one that cannot raises CPython's TypeError, word for word,
# including the suggestion for a misspelled keyword.
#
# WHY THIS IS RUN: the class a value names is known only at run time, and so
# is the TypeError. One subclass with another `__init__` refused the call for
# every class ("missing required argument 'extra' for inlined class method"),
# which is unittest's loader building a case per test name.
class TestCase:
    def __init__(self, methodName: str = "runTest") -> None:
        self.methodName = methodName


class Holder(TestCase):
    def __init__(self, description: str, extra: int) -> None:
        super().__init__()
        self.description = description


class Keyed(TestCase):
    def __init__(self, *, key: str) -> None:
        super().__init__()
        self.key = key


class Spelled(TestCase):
    def __init__(self, method_name: str = "x", methodname: str = "y") -> None:
        super().__init__(method_name + methodname)


class MyTests(TestCase):
    def test_a(self) -> None:
        pass


def load(cls: type[TestCase], names: list[str]) -> list[TestCase]:
    return [cls(n) for n in names]


def make(cls: type[TestCase]) -> TestCase:
    return cls(methodName="x")


print([t.methodName for t in load(MyTests, ["test_a", "test_b"])])
print(Holder("d", 1).description, Keyed(key="k").key, Spelled().methodName)
for cls in [Holder, Keyed, Spelled]:
    try:
        print([t.methodName for t in load(cls, ["x"])])
    except TypeError as e:
        print("TypeError:", e)
    try:
        print(make(cls).methodName)
    except TypeError as e:
        print("TypeError:", e)
print(make(MyTests).methodName)
