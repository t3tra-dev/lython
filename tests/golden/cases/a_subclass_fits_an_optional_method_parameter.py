# WHAT: a method whose parameter is a class or None, as unittest's
# `TestCase.run(result=None)`, called with an instance of a subclass, with
# the class itself, and with nothing; the method reads what it was given.
#
# WHY THIS IS RUN: the method body runs against the argument's own class. A
# subclass instance was refused for a `Base | None` parameter of a method
# ("argument 'result' of 'run' is declared ... and this call gives it ...")
# while a `Base` parameter, or a free function's `Base | None`, took it.
class Result:
    def __init__(self) -> None:
        self.runs = 0

    def describe(self) -> str:
        return "result " + str(self.runs)


class TextResult(Result):
    def describe(self) -> str:
        return "text " + str(self.runs)


class Case:
    def run(self, result: Result | None = None) -> str:
        if result is None:
            result = Result()
        result.runs += 1
        return result.describe()


c = Case()
t = TextResult()
print(c.run(t), c.run(t), c.run(Result()), c.run())
