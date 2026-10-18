# WHAT: `map(cls, names)` with `cls` a class value held in a parameter builds
# one instance of the class the value names per element -- unittest's loader
# spells `self.suiteClass(map(testCaseClass, testCaseNames))` -- consumed by a
# comprehension, by list() and by a for loop, with filter beside it.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which class each element
# is constructed as is the run-time class of the value. The lazy iterator's
# body is a def of its own, where the parameter's spelling named nothing
# ("unresolved runtime binding 'cls'").
class Case:
    def __init__(self, methodName: str = "runTest") -> None:
        self.name = methodName


class MyCase(Case):
    pass


def load(cls: type[Case], names: list[str]) -> list[Case]:
    return list(map(cls, names))


def kinds(cls: type[Case], names: list[str]) -> list[str]:
    return [type(c).__name__ + ":" + c.name for c in map(cls, names)]


print([c.name for c in load(MyCase, ["test_a", "test_b"])])
print(kinds(MyCase, ["x"]), kinds(Case, ["y", "z"]))
for c in map(MyCase, ["w"]):
    print(type(c).__name__, c.name)
print([n for n in filter(lambda n: n.startswith("test"), ["test_a", "helper"])])
