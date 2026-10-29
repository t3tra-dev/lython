# A tuple of classes held in a name, a field or a parameter is any of its
# classes, as the same tuple written out is: in isinstance, issubclass and an
# except clause, member by member when the members differ and walked when its
# length is known only at run time. A loop over one binds each class in turn,
# which can be tested against, named and called. The value tested and the
# tuple are each evaluated once. A member of a tuple is read by position
# whatever its members are: classes, instances of one base, functions, or
# tuples of one length; tuples of different lengths in one list are tuples
# of any length, each read as itself. A class from a loop over a tuple of a
# class and its subclass is a class value of the base, and two of them are
# the same object exactly when they are the same class.
import unittest

NUMBER = (int, float)
LOOKUP_ERRORS = (KeyError, IndexError)


class Animal:
    def speak(self) -> str:
        return "..."


class Dog(Animal):
    def speak(self) -> str:
        return "woof"


class Cat(Animal):
    def speak(self) -> str:
        return "meow"


class Parsers:
    errors = (ValueError, TypeError)


calls: list[str] = []


def subject(v: object) -> object:
    calls.append("subject")
    return v


def classes() -> tuple[type, ...]:
    calls.append("classes")
    return (int, str)


def pair() -> tuple[type[int], type[str]]:
    calls.append("pair")
    return (int, str)


def kind(x: int | float | str) -> str:
    if isinstance(x, NUMBER):
        return "number"
    return "text"


def lookup(d: dict[str, int], xs: list[int], k: str, i: int) -> str:
    try:
        return str(d[k] + xs[i])
    except LOOKUP_ERRORS as e:
        return "missing " + type(e).__name__


def parse(s: str) -> int:
    try:
        return int(s)
    except Parsers.errors:
        return -1


def one(x: int) -> int:
    return x + 1


def two(x: int, y: int = 2) -> int:
    return x * y


class HeldClassTupleTest(unittest.TestCase):
    def test_isinstance_with_a_held_tuple(self):
        self.assertEqual([kind(1), kind(1.5), kind("s")],
                         ["number", "number", "text"])
        self.assertTrue(isinstance(True, NUMBER))
        self.assertFalse(isinstance("1", NUMBER))

    def test_issubclass_with_a_held_tuple(self):
        self.assertTrue(issubclass(bool, NUMBER))
        self.assertFalse(issubclass(str, NUMBER))

    def test_the_subject_and_the_tuple_run_once(self):
        calls.clear()
        self.assertTrue(isinstance(subject(1), classes()))
        self.assertEqual(calls, ["subject", "classes"])
        calls.clear()
        self.assertFalse(isinstance(subject(2.5), pair()))
        self.assertEqual(calls, ["subject", "pair"])

    def test_except_with_a_held_tuple(self):
        self.assertEqual(lookup({"a": 1}, [1], "a", 0), "2")
        self.assertEqual(lookup({}, [1], "a", 0), "missing KeyError")
        self.assertEqual(lookup({"a": 1}, [], "a", 3), "missing IndexError")
        self.assertEqual([parse("3"), parse("x")], [3, -1])

    def test_a_loop_binds_each_class(self):
        names = []
        tested = []
        for c in (int, str, float):
            names.append(c.__name__)
            tested.append(isinstance(True, c))
            tested.append(issubclass(bool, c))
        self.assertEqual(names, ["int", "str", "float"])
        self.assertEqual(tested, [True, True, False, False, False, False])
        self.assertEqual([c("5") for c in (int, str)], [5, "5"])
        self.assertEqual([cls().speak() for cls in (Dog, Cat)],
                         ["woof", "meow"])

    def test_a_member_read_by_position(self):
        classes_held = (int, str)
        animals = (Dog(), Cat())
        functions = (one, two)
        pairs = ((1, 1), ("s", "t"))
        self.assertEqual(classes_held[1].__name__, "str")
        self.assertEqual(animals[0].speak() + animals[1].speak(), "woofmeow")
        self.assertEqual(functions[0](3) + functions[1](3), 10)
        self.assertEqual(pairs[0][1] + 1, 2)
        self.assertEqual(pairs[1][0], "s")

    def test_tuples_of_different_lengths_in_one_list(self):
        rows = [(1,), (1, 2), (3, 4, 5)]
        self.assertEqual([(len(r), sum(r), r[0]) for r in rows],
                         [(1, 1, 1), (2, 3, 1), (3, 12, 3)])
        first, *rest = rows[2]
        self.assertEqual((first, rest), (3, [4, 5]))


    def test_a_class_from_the_loop_meets_its_base(self):
        def named(cls: type[Animal]) -> str:
            return cls.__name__

        self.assertEqual([named(c) for c in (Animal, Dog)], ["Animal", "Dog"])
        kinds: list[type[Dog] | type[Cat]] = [Dog, Cat, Dog]
        self.assertEqual([kinds[0] is kinds[2], kinds[0] is kinds[1],
                          kinds[1] is not kinds[2]], [True, False, True])


if __name__ == "__main__":
    unittest.main()
