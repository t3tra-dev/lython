# A method of the program reads the program's module-level names, also where
# code of another module is what calls it -- unittest's runner calling setUp
# and a test, a base class of another module calling the hook a subclass
# overrides -- and not that module's names: unittest's case module binds `time`
# to the module and DIFF_OMITTED to a str, and the base class's methods have
# locals spelled `out` and `tag`.
import unittest
from time import time

import hook_base

LIMIT = 7
DIFF_OMITTED = 42
out = "program"
tag = 1
log: list[str] = []


def helper(n: int) -> int:
    return n * LIMIT


class Point:
    def __init__(self, x: int) -> None:
        self.x = x


class Page(hook_base.Template):
    def hook(self) -> str:
        return out

    def measure(self) -> int:
        return tag * 100


RENDERED = (Page().render(), Page().total())


class RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.p = Point(LIMIT)
        self.facts = (time() > 0.0, DIFF_OMITTED + 1)
        log.append("setUp")
        self.addCleanup(log.append, "cleanup")

    def test_1_reads_the_programs_names(self) -> None:
        self.assertEqual((helper(2), self.p.x, self.facts), (14, 7, (True, 43)))
        log.append("first")

    def test_2_sees_the_first_tests_cleanup(self) -> None:
        self.assertEqual(log, ["setUp", "first", "cleanup", "setUp"])


class HookTest(unittest.TestCase):
    def test_called_from_module_level(self) -> None:
        self.assertEqual(RENDERED, ("program/base", 110))

    def test_called_from_a_test(self) -> None:
        self.assertEqual((Page().render(), Page().total()), ("program/base", 110))


if __name__ == "__main__":
    unittest.main()
