# A value whose type is a union of members -- `str | list[str]`, `int | float`
# -- is iterated, converted, searched, sliced and read as the member it holds:
# a `for` loop and every comprehension or builtin that iterates it, the
# builtins that convert or measure one value, `in`, a slice, and an attribute
# its members answer differently. The loop target outlives the loop as
# whichever member's element it was last bound to. A conditional expression
# between two union values is whichever of them its test picks.
#
# A result holding union values is compared by its repr, which tells 'a', 1
# and 1.0 apart as the values themselves would.
import unittest


class Bag:
    def __init__(self, xs: list[int]) -> None:
        self.xs = xs

    def __iter__(self):
        return iter(self.xs)


class Counted:
    def __init__(self) -> None:
        self.n = 1


class Named:
    def __init__(self) -> None:
        self.n = "q"


def counted_or_named(c: bool):
    return Counted() if c else Named()


def pick(n: int):
    if n == 0:
        return "xyz"
    if n == 1:
        return [10, 20, 30]
    return (7, 8)


def walk(n: int) -> str:
    out = []
    u = pick(n)
    for x in u:
        if x == 20 or x == "y":
            out.append("brk")
            break
        out.append(str(x))
    else:
        out.append("else")
    out.append("last=" + str(x))
    return ",".join(out)


def text_or_words(c: bool):
    return "abc" if c else ["a", "z"]


def int_or_float(c: bool):
    return 2.5 if c else -3


def closures(c: bool):
    u = "ab" if c else [1, 2]
    fs = []
    for x in u:
        fs.append(lambda: x)
    return [f() for f in fs]


def each(c: bool):
    u = "ab" if c else [1, 2]
    for x in u:
        yield x


def weighted(c: bool) -> int:
    u = [1, 2, 3] if c else (4, 5)
    t = 0
    for i, x in enumerate(u):
        t += i * x
    return t


def pairs(c: bool, d: bool):
    u = "ab" if c else [1, 2]
    w = "cd" if d else [3]
    r = []
    for a in u:
        for b in w:
            r.append((a, b))
    return r


def consumers(c: bool):
    u = text_or_words(c)
    return (sorted(u), list(u), tuple(u), sorted(set(u)), max(u), min(u),
            all(u), [x.upper() for x in u], {x: 1 for x in u},
            list(enumerate(u)), list(iter(u)))


order: list[str] = []


def note(tag: str, v: int) -> int:
    order.append(tag)
    return v


def conversions(c: bool):
    v = int_or_float(c)
    return (abs(v), round(v), int(v), float(v), divmod(note("a", 7), v),
            hash(v) == hash(2.5 if c else -3))


def membership(c: bool):
    u = text_or_words(c)
    return "a" in u, "q" not in u, u[1:], u[::-1]


def last_of(xs: list[int | str]):
    for x in xs:
        pass
    return x


def longest(c: bool):
    u = "ab" if c else "cde"
    return max(len(x) for x in [u, u + u])


def from_a_class(c: bool):
    u = Bag([5, 6]) if c else "hi"
    return [x for x in u], sum(1 for _ in u)


class UnionIterationTest(unittest.TestCase):
    def test_a_loop_runs_as_the_member_it_holds(self):
        self.assertEqual(walk(0), "x,brk,last=y")
        self.assertEqual(walk(1), "10,brk,last=20")
        self.assertEqual(walk(2), "7,8,else,last=8")

    def test_a_closure_in_the_loop_reads_the_last_target(self):
        self.assertEqual(repr(closures(True)), "['b', 'b']")
        self.assertEqual(repr(closures(False)), "[2, 2]")

    def test_a_generator_loops_over_a_union(self):
        self.assertEqual(list(each(True)), ["a", "b"])
        self.assertEqual(list(each(False)), [1, 2])

    def test_enumerate_and_nested_loops(self):
        self.assertEqual(weighted(True), 8)
        self.assertEqual(weighted(False), 5)
        self.assertEqual(pairs(True, False), [("a", 3), ("b", 3)])
        self.assertEqual(pairs(False, True),
                         [(1, "c"), (1, "d"), (2, "c"), (2, "d")])

    def test_what_iterates_a_union(self):
        self.assertEqual(
            repr(consumers(True)),
            "(['a', 'b', 'c'], ['a', 'b', 'c'], ('a', 'b', 'c'), "
            "['a', 'b', 'c'], 'c', 'a', True, ['A', 'B', 'C'], "
            "{'a': 1, 'b': 1, 'c': 1}, [(0, 'a'), (1, 'b'), (2, 'c')], "
            "['a', 'b', 'c'])")
        self.assertEqual(
            repr(consumers(False)),
            "(['a', 'z'], ['a', 'z'], ('a', 'z'), ['a', 'z'], 'z', 'a', True, "
            "['A', 'Z'], {'a': 1, 'z': 1}, [(0, 'a'), (1, 'z')], ['a', 'z'])")

    def test_a_class_member_iterates_by_its_iter(self):
        self.assertEqual(repr(from_a_class(True)), "([5, 6], 2)")
        self.assertEqual(repr(from_a_class(False)), "(['h', 'i'], 2)")

    def test_builtins_convert_the_member_it_holds(self):
        order.clear()
        self.assertEqual(repr(conversions(True)),
                         "(2.5, 2, 2, 2.5, (2.0, 2.0), True)")
        self.assertEqual(repr(conversions(False)),
                         "(3, -3, -3, -3.0, (-3, -2), True)")
        self.assertEqual(order, ["a", "a"])

    def test_in_and_a_slice(self):
        self.assertEqual(membership(True), (True, True, "bc", "cba"))
        self.assertEqual(membership(False), (True, True, ["z"], ["z", "a"]))

    def test_a_target_of_union_elements_outlives_the_loop(self):
        self.assertEqual(last_of([1, "a"]), "a")
        self.assertEqual(last_of(["a", 1]), 1)

    def test_an_attribute_of_the_member_it_holds(self):
        self.assertEqual(
            repr([counted_or_named(True).n, counted_or_named(False).n]),
            "[1, 'q']")
        v = int_or_float(True)
        w = int_or_float(False)
        self.assertEqual(repr((v.real, v.imag, w.real, w.imag)),
                         "(2.5, 0.0, -3, 0)")

    def test_a_generator_expression_over_a_local_types_the_return(self):
        self.assertEqual(longest(True), 4)
        self.assertEqual(longest(False), 6)


    def test_a_conditional_expression_between_union_values(self):
        def pick(a: str | None, b: str | None, c: bool) -> str | None:
            return a if c else b

        def either(a: int | str, b: int | str, c: bool) -> int | str:
            return a if c else b

        self.assertEqual([pick("x", None, True), pick("x", None, False)],
                         ["x", None])
        self.assertEqual(repr([either(1, "s", True), either(1, "s", False)]),
                         "[1, 's']")


if __name__ == "__main__":
    unittest.main()
