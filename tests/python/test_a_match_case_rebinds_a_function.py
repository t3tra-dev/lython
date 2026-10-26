# A name bound before a `match` and again in one of its cases holds what the
# case gave it -- a function as well as an int, in a function and in the
# module, where a function body reads it too.
import unittest


def first(x: int) -> str:
    return "first " + str(x)


def second(x: int) -> str:
    return "second " + str(x)


def choose(n: int) -> str:
    chosen = first
    match n:
        case 1:
            chosen = second
        case _:
            pass
    return chosen(n)


def choose_without_default(n: int) -> str:
    chosen = first
    match n:
        case 1:
            chosen = second
    return chosen(n)


module_chosen = first
module_count = 1
match len("a"):
    case 1:
        module_chosen = second
        module_count = 2


def read_module_names() -> str:
    return module_chosen(module_count)


class MatchRebindsAFunctionTest(unittest.TestCase):

    def test_in_a_function(self) -> None:
        self.assertEqual(choose(1), "second 1")
        self.assertEqual(choose(2), "first 2")
        self.assertEqual(choose_without_default(1), "second 1")
        self.assertEqual(choose_without_default(2), "first 2")

    def test_in_the_module(self) -> None:
        self.assertEqual(module_chosen(3), "second 3")
        self.assertEqual(module_count, 2)

    def test_a_function_reads_the_module_names(self) -> None:
        self.assertEqual(read_module_names(), "second 2")


if __name__ == "__main__":
    unittest.main()
