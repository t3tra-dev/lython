# `isinstance(x, cls)` and `case cls():` with `cls` a class value -- a
# parameter, a loop over classes, a field -- test the class the value holds,
# which may be any class under its static one; only that the value is an
# instance of the static class follows, and only where the test held.
import unittest


class Base:
    def name(self) -> str:
        return "base"


class A(Base):
    def name(self) -> str:
        return "A"


class B(Base):
    pass


def is_one(e: Exception, cls: type[Exception]) -> bool:
    if isinstance(e, cls):
        return True
    return False


def is_not_one(e: Exception, cls: type[Exception]) -> bool:
    if not isinstance(e, cls):
        return True
    return False


def first_match(e: Exception) -> str:
    for i, cls in enumerate([KeyError, LookupError, ValueError, Exception]):
        if isinstance(e, cls):
            return f"{i}:{cls.__name__}"
    return "none"


def picked(items: list[Base], cls: type[Base]) -> int:
    return len([x for x in items if isinstance(x, cls)])


def named_or_none(x: Base | None, cls: type[Base]) -> str:
    if isinstance(x, cls):
        return "yes:" + x.name()
    if x is None:
        return "none"
    return x.name()


def asserted(e: Exception, cls: type[Exception]) -> str:
    try:
        assert isinstance(e, cls)
    except AssertionError:
        return "failed"
    return "held"


def of_object(o: object, cls: type) -> bool:
    if isinstance(o, cls):
        return True
    return False


def in_a_tuple(o: object, cls: type[Base]) -> bool:
    if isinstance(o, (cls, int)):
        return True
    return False


def matched(x: Base, cls: type[Base]) -> str:
    match x:
        case cls():
            return "match"
        case _:
            return "no"


def matched_optional(x: Base | None, cls: type[Base]) -> str:
    match x:
        case cls():
            return "hit"
        case None:
            return "none"
        case _:
            return "other"


class ClassValueTestTest(unittest.TestCase):
    def test_an_if_tests_the_held_class(self):
        self.assertFalse(is_one(ValueError(), KeyError))
        self.assertTrue(is_one(KeyError(), LookupError))
        self.assertTrue(is_not_one(ValueError(), KeyError))
        self.assertFalse(is_not_one(KeyError(), KeyError))

    def test_a_loop_over_classes_finds_the_first_that_holds(self):
        self.assertEqual(
            [first_match(KeyError()), first_match(IndexError()),
             first_match(ValueError()), first_match(RuntimeError())],
            ["0:KeyError", "1:LookupError", "2:ValueError", "3:Exception"])

    def test_a_comprehension_filter(self):
        items: list[Base] = [A(), B(), A()]
        self.assertEqual([picked(items, A), picked(items, B),
                          picked(items, Base)], [2, 1, 3])

    def test_the_true_side_is_the_bound_and_the_false_side_is_not_narrowed(
            self):
        self.assertEqual([named_or_none(A(), A), named_or_none(Base(), A),
                          named_or_none(None, Base)],
                         ["yes:A", "base", "none"])

    def test_an_assert(self):
        self.assertEqual([asserted(KeyError(), LookupError),
                          asserted(ValueError(), KeyError)],
                         ["held", "failed"])

    def test_an_object_subject_and_a_tuple_target(self):
        self.assertEqual([of_object(1, str), of_object("s", str)],
                         [False, True])
        self.assertEqual([in_a_tuple(B(), A), in_a_tuple(A(), A),
                          in_a_tuple(3, A)], [False, True, True])

    def test_a_class_pattern(self):
        self.assertEqual([matched(A(), B), matched(B(), B), matched(B(), Base)],
                         ["no", "match", "match"])
        self.assertEqual([matched_optional(A(), B), matched_optional(B(), B),
                          matched_optional(None, A)], ["other", "hit", "none"])


if __name__ == "__main__":
    unittest.main()
