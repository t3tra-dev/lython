# WHAT: a runner calls a test method by a computed name, and one subclass's
# `__init__` takes an argument the base's does not. The lookup can reach
# `__init__` on any class -- the subclass's raises the TypeError CPython
# raises for a call without that argument, and the base's re-runs its own
# initializer -- while every ordinary test method is called as before.
#
# WHY THIS IS RUN: which `__init__` a computed name reaches is the receiver's
# class at run time, and the TypeError is raised then. The program was
# refused at compile time ("missing required argument 'description' for
# inlined class method") although no test ever names `__init__`.
class TestCase:
    def __init__(self, methodName: str = "runTest") -> None:
        self._testMethodName = methodName

    def run(self) -> None:
        getattr(self, self._testMethodName)()


class Holder(TestCase):
    def __init__(self, description: str) -> None:
        super().__init__()
        self.description = description


class MyTests(TestCase):
    def test_a(self) -> None:
        print("test_a ran")


MyTests("test_a").run()
print(Holder("d").description)
h = Holder("e")
h._testMethodName = "__init__"
try:
    h.run()
except TypeError as e:
    print("TypeError:", e)
m = MyTests("__init__")
m.run()
print(m._testMethodName)
