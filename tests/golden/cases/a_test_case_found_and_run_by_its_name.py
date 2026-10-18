# WHAT: test classes declared in this module, subclassing a TestCase from the
# module before: `dir` lists each class's test methods, a class value builds
# one case per name, and the base's `run` -- emitted before this module's
# classes are -- calls the method by its computed name; a test that fails, one
# that errors, a name no class declares, and a name only a sibling declares.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which method a computed
# name reaches is decided by the case's class at run time. The lookup was
# refused ("getattr() needs a literal attribute name"), and `dir` was an
# unresolved name.
from a_module_of_a_test_framework import TestCase, load, test_names


class MathTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.base = 10

    def test_add(self) -> None:
        assert self.base + 1 == 11

    def test_fail(self) -> None:
        assert self.base == 11, "base is " + str(self.base)

    def test_error(self) -> None:
        {}["missing"]

    def helper(self) -> int:
        return self.base


class StrTest(TestCase):
    def test_upper(self) -> None:
        assert "a".upper() == "A"


print(test_names(MathTest), test_names(StrTest))
for cls in [MathTest, StrTest]:
    for case in load(cls):
        print(type(case).__name__, case._testMethodName, "->", case.run(),
              case.log)
for case in [StrTest("test_add"), MathTest("nope"), MathTest("helper")]:
    print(type(case).__name__, case._testMethodName, "->", case.run(),
          case.log)
