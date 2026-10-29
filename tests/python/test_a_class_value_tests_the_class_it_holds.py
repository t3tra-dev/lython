# `isinstance(x, cls)` and `case cls():` with `cls` a class value -- a
# parameter, a loop over classes, a field -- test the class the value holds,
# which may be any class under its static one; only that the value is an
# instance of the static class follows, and only where the test held: a
# class pattern that fails leaves the subject's later patterns refutable. A
# class value that may hold one of several classes proves one of them. A
# class pattern's keyword names any attribute: a class attribute, a property,
# an exception's args. A class pattern over a class value of several classes
# tests the one it holds; classes with no common base chosen into one name
# are a class value of either, which can be called and passed on.
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


class Other:
    def __init__(self) -> None:
        self.other = "other-field"


def after_a_failed_pattern(x: Base | Other, cls: type[Base]) -> str:
    match x:
        case cls():
            return "cls " + x.name()
        case Other():
            return "Other " + x.other
        case _:
            return "rest " + type(x).__name__


class Tagged:
    def __init__(self, v: str) -> None:
        self.v = v


class Marked:
    def __init__(self, v: str) -> None:
        self.v = v


def read_when_one_of(o: Tagged | Marked | Other,
                     k: type[Tagged] | type[Marked]) -> str:
    if isinstance(o, k):
        return o.v
    return "miss"


class Labelled:
    tag = "t"

    def __init__(self, x: int) -> None:
        self.x = x

    @property
    def doubled(self) -> int:
        return self.x * 2


def by_attributes(v: Labelled, cls: type[Labelled]) -> str:
    match v:
        case cls(tag=t, x=n):
            return f"{t} {n}"
    return "no"


def by_property(v: Labelled) -> str:
    match v:
        case Labelled(doubled=4):
            return "four"
        case Labelled(doubled=d, tag="t"):
            return f"d={d}"
    return "no"


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

    def test_a_failed_class_pattern_keeps_the_later_ones_refutable(self):
        self.assertEqual([after_a_failed_pattern(A(), B),
                          after_a_failed_pattern(B(), B),
                          after_a_failed_pattern(Other(), A)],
                         ["rest A", "cls base", "Other other-field"])

    def test_a_class_value_of_several_classes_proves_one_of_them(self):
        self.assertEqual([read_when_one_of(Tagged("t"), Tagged),
                          read_when_one_of(Marked("m"), Tagged),
                          read_when_one_of(Marked("m"), Marked),
                          read_when_one_of(Other(), Marked)],
                         ["t", "miss", "m", "miss"])


    def test_a_class_pattern_keyword_names_any_attribute(self):
        self.assertEqual([by_attributes(Labelled(1), Labelled),
                          by_property(Labelled(2)), by_property(Labelled(3))],
                         ["t 1", "four", "d=6"])
        try:
            raise ValueError("boom", 2)
        except ValueError as e:
            match e:
                case ValueError(args=a):
                    self.assertEqual(a, ("boom", 2))


    def test_a_class_pattern_over_several_classes(self):
        def route(o: Tagged | Marked | Other,
                  k: type[Tagged] | type[Marked]) -> str:
            match o:
                case k(v=found):
                    return "hit " + found
                case Other():
                    return "other"
                case _:
                    return "miss " + type(o).__name__

        self.assertEqual([route(Tagged("t"), Tagged), route(Marked("m"), Tagged),
                          route(Marked("m"), Marked), route(Other(), Marked)],
                         ["hit t", "miss Marked", "hit m", "other"])

    def test_classes_with_no_common_base_are_either(self):
        def named(k: type[Tagged] | type[Marked]) -> str:
            return k.__name__ + ":" + k("x").v

        def pick(first: bool) -> str:
            k = Tagged if first else Marked
            return named(k)

        self.assertEqual([pick(True), pick(False)], ["Tagged:x", "Marked:x"])
        convert = int if len(str(pick(True))) > 3 else str
        self.assertEqual((convert("5"), convert.__name__), (5, "int"))
        self.assertEqual([kind("3") for kind in [int, float]], [3, 3.0])


if __name__ == "__main__":
    unittest.main()
