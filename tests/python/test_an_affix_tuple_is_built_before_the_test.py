# `s.startswith(t)` and `s.endswith(t)` with a tuple are any of its members,
# whether the tuple is written in the call or held in a name or a parameter.
# The tuple is built before the test runs, as in CPython: every member is
# evaluated even when an earlier one matches, and `start` and `end` are
# evaluated once, after the tuple.
import unittest

order: list[str] = []


def tag(name: str, v: str) -> str:
    order.append(name)
    return v


def at(name: str, v: int) -> int:
    order.append(name)
    return v


def has_scheme(url: str, schemes: tuple[str, ...]) -> bool:
    return url.startswith(schemes)


class AffixTupleTest(unittest.TestCase):
    def test_every_member_is_evaluated(self):
        order.clear()
        self.assertTrue("abc".startswith((tag("p1", "a"), tag("p2", "z"))))
        self.assertEqual(order, ["p1", "p2"])

    def test_the_bounds_are_evaluated_once_after_the_tuple(self):
        order.clear()
        self.assertTrue("abc".startswith(("x", "y", "b"), at("start", 1)))
        self.assertEqual(order, ["start"])
        order.clear()
        self.assertTrue("abc".endswith((tag("s1", "q"), tag("s2", "c")),
                                       at("st", 0), at("en", 3)))
        self.assertEqual(order, ["s1", "s2", "st", "en"])

    def test_a_held_tuple(self):
        prefixes = ("a", "h")
        self.assertTrue("https".startswith(prefixes))
        self.assertFalse("https".startswith(prefixes, 1))
        self.assertTrue(has_scheme("https://x", ("http://", "https://")))
        self.assertFalse(has_scheme("ftp://x", ("http://", "https://")))
        self.assertFalse(has_scheme("x", ()))
        self.assertTrue(b"abc".endswith((b"bc",), 0, 3))

    def test_an_empty_tuple_still_evaluates_the_bounds(self):
        order.clear()
        self.assertFalse("abc".startswith((), at("start", 0)))
        self.assertEqual(order, ["start"])


if __name__ == "__main__":
    unittest.main()
