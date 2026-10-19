# WHAT: `hasattr(cls, name)` for a class value whose static class lacks the
# name and whose subclasses declare it -- as a method, as a class attribute
# -- or inherit it, and for an instance of one; inside an `and` with another
# test, the way unittest's loader asks for `runTest`.
#
# WHY THIS IS RUN: the answer is the class the value names at run time. It
# was refused ("'WithRunTest' derives from it and may define that
# attribute"), and `not names and hasattr(cls, 'runTest')` was an `and` with
# no representable result.
class TestCase:
    pass


class WithRunTest(TestCase):
    def runTest(self) -> None:
        pass


class Inherits(WithRunTest):
    pass


class Flagged(TestCase):
    runTest = "a string"


def names_for(cls: type[TestCase], names: list[str]) -> list[str]:
    if not names and hasattr(cls, 'runTest'):
        return ['runTest']
    return names


def has(test: TestCase) -> bool:
    return hasattr(test, 'runTest')


classes: list[type[TestCase]] = [TestCase, WithRunTest, Inherits, Flagged]
print([hasattr(c, 'runTest') for c in classes])
print([names_for(c, []) for c in classes], names_for(WithRunTest, ["x"]))
print([has(c()) for c in classes])
