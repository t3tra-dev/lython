# A generic method, and every method of a generic class, has one default per
# parameter, evaluated once when the class statement runs, as CPython
# evaluates a def's defaults. Each call that leaves the argument out reads
# that one object, whatever the call solves the type parameters to, and
# wherever the call is: at module level after the class, in a method, or in
# an arm of `getattr(self, name)()`.
import unittest

made: list[str] = []


def make() -> list[int]:
    made.append("made")
    return [0]


class Grower:
    def grow[T](self, v: T, into: list[int] = make()) -> int:
        into.append(len(into))
        return len(into)

    def note[T](self, v: T, seen: list[str] = []) -> list[str]:
        seen.append(type(v).__name__)
        return seen


class Box[T]:
    def __init__(self, v: T) -> None:
        self.v = v

    def note(self, seen: list[str] = []) -> list[str]:
        seen.append(repr(self.v))
        return seen


class Tally:
    def count[T](self, v: T, seen: list[str] = []) -> int:
        seen.append(repr(v))
        return len(seen)


picked: list[str] = []


class Case:
    def __init__(self, name: str) -> None:
        self.name = name

    def pick[E: BaseException](self, expected: type[E] = BaseException) -> None:
        picked.append(expected.__name__)

    def named(self) -> None:
        self.pick(ValueError)

    def run(self) -> None:
        getattr(self, self.name)()


grower = Grower()
GROWN = (grower.grow(1), grower.grow("x"), grower.grow(1.5))
MADE = list(made)
NOTED = grower.note(1) is grower.note("a")
SEEN = list(grower.note(2.5))
BOXES = (list(Box(1).note()), list(Box("s").note()))
Case("pick").run()
Case("named").run()
Case("x").pick()
PICKED = list(picked)


class SharedDefaultTest(unittest.TestCase):
    def test_module_level_calls_after_the_class_share_one_default(self):
        self.assertEqual(GROWN, (2, 3, 4))
        self.assertEqual(MADE, ["made"])

    def test_every_solution_reads_the_same_object(self):
        self.assertTrue(NOTED)
        self.assertEqual(SEEN, ["int", "str", "float"])

    def test_a_generic_class_shares_its_method_default(self):
        self.assertEqual(BOXES, (["1"], ["1", "'s'"]))

    def test_calls_in_a_method(self):
        tally = Tally()
        self.assertEqual(tally.count(1), 1)
        self.assertEqual(tally.count("a"), 2)
        self.assertEqual(Tally().count(2.5), 3)

    def test_an_arm_of_a_computed_call(self):
        self.assertEqual(PICKED, ["BaseException", "ValueError", "BaseException"])


if __name__ == "__main__":
    unittest.main()
