# A union given to a call that takes one value of one kind runs the call as
# the member the union holds: a builtin type's method taking it through a
# protocol (`sep.join(u)`, `xs.extend(u)`, `s.update(u)`), the iteration
# builtins that walk it by index (`reversed`, `zip`, `map`, `iter`), a
# constructor (`frozenset(u)`) and a format specification. The receiver and the
# arguments are evaluated once, in order, before the member is asked. A
# function with no return annotation returns what `map` over two sequences
# gives. A method parameter declared as a base takes a union of the base and
# its subclasses, and a list display whose elements fit a union element
# type, a subclass instance among them, is built as that list.
import unittest

order: list[str] = []


def note(tag: str, v: str) -> str:
    order.append(tag)
    return v


def words(c: bool):
    return ("a", "b") if c else ["x", "y", "z"]


def text_or_words(c: bool):
    return "abc" if c else ["a", "z"]


def nums(c: bool):
    return (1, 2) if c else [3, 4]


def num(c: bool):
    return 1.5 if c else 3


def extended(c: bool) -> list[int]:
    xs = [0]
    xs.extend(nums(c))
    return xs


def updated(c: bool) -> list[int]:
    s = {9}
    s.update(nums(c))
    return sorted(s)


def walked(c: bool):
    u = text_or_words(c)
    backwards = [x for x in reversed(u)]
    return (list(reversed(u)), backwards, list(zip([10, 20], u)),
            list(zip(u, u)), list(map(lambda a, b: a + b, ["p", "q"], u)))


def stepped(c: bool):
    it = iter(text_or_words(c))
    first = next(it)
    return first, next(it)


class UnionArgumentTest(unittest.TestCase):
    def test_a_method_taking_an_iterable(self):
        order.clear()
        self.assertEqual(note("sep", "-").join(words(True)), "a-b")
        self.assertEqual(note("sep", "-").join(words(False)), "x-y-z")
        self.assertEqual(order, ["sep", "sep"])
        self.assertEqual(extended(True), [0, 1, 2])
        self.assertEqual(extended(False), [0, 3, 4])
        self.assertEqual(updated(True), [1, 2, 9])
        self.assertEqual(updated(False), [3, 4, 9])

    def test_the_iteration_builtins_walk_it_by_index(self):
        self.assertEqual(
            repr(walked(True)),
            "(['c', 'b', 'a'], ['c', 'b', 'a'], [(10, 'a'), (20, 'b')], "
            "[('a', 'a'), ('b', 'b'), ('c', 'c')], ['pa', 'qb'])")
        self.assertEqual(
            repr(walked(False)),
            "(['z', 'a'], ['z', 'a'], [(10, 'a'), (20, 'z')], "
            "[('a', 'a'), ('z', 'z')], ['pa', 'qz'])")
        self.assertEqual(stepped(True), ("a", "b"))
        self.assertEqual(stepped(False), ("a", "z"))

    def test_a_constructor_and_a_format(self):
        self.assertEqual(sorted(frozenset(nums(True))), [1, 2])
        self.assertEqual(sorted(frozenset(nums(False))), [3, 4])
        self.assertEqual(format(num(True), ">6"), "   1.5")
        self.assertEqual(f"{num(False):>6}|{num(False):06.2f}",
                         "     3|003.00")


    def test_a_method_parameter_takes_a_union_of_its_subclasses(self):
        class Event:
            def __init__(self, src: str) -> None:
                self.src = src

        class Stop(Event):
            pass

        class Bus:
            def emit(self, e: Event) -> str:
                return e.src + ":" + type(e).__name__

        bus = Bus()
        self.assertEqual([bus.emit(e) for e in (Event("a"), Stop("b"))],
                         ["a:Event", "b:Stop"])


    def test_a_display_with_a_subclass_meets_a_union_element(self):
        class Left:
            pass

        class SubLeft(Left):
            pass

        class Right:
            pass

        def names(xs: list[Left | Right]) -> list[str]:
            return [type(x).__name__ for x in xs]

        self.assertEqual(names([Left(), Right(), SubLeft()]),
                         ["Left", "Right", "SubLeft"])


if __name__ == "__main__":
    unittest.main()
