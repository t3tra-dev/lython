# WHAT: sixteen tests -- twelve pass, three are skipped, one fails as its
# expectedFailure says -- covering equality of ints, strings, lists, dicts and
# sets, identity, None, almost-equal, ordering, assertRaises in both forms,
# class and method fixtures, the three skip decorators and skipTest, run by
# the prototype unittest next to this file. Under python3.14 and under
# `lyc jit` the stdout, the stderr (but for the time) and the exit status are
# the same.
import unittest


def double(x: int) -> int:
    return 2 * x


def divide(a: int, b: int) -> int:
    return a // b


def parse(text: str) -> int:
    return int(text)


class ArithmeticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = [1, 2, 3]

    def test_double(self) -> None:
        self.assertEqual(double(2), 4)
        self.assertNotEqual(double(2), 5)

    def test_strings(self) -> None:
        self.assertEqual("a" + "b", "ab")
        self.assertTrue("abc".startswith("a"))
        self.assertFalse("abc".startswith("b"))

    def test_lists(self) -> None:
        self.assertEqual(self.values, [1, 2, 3])
        self.assertListEqual([double(v) for v in self.values], [2, 4, 6])
        self.assertIn(2, self.values)
        self.assertNotIn(7, self.values)

    def test_dicts_and_sets(self) -> None:
        self.assertDictEqual({"a": 1}, {"a": 1})
        # WORKAROUND(ut_bug_set_display_solves_no_type_parameter): set displays do not solve a generic method's T.
        a: set[int] = {1, 2}
        b: set[int] = {2, 1}
        self.assertSetEqual(a, b)

    def test_none_and_identity(self) -> None:
        x: int | None = None
        self.assertIsNone(x)
        self.assertIsNotNone(self.values)
        self.assertIs(self.values, self.values)

    def test_almost(self) -> None:
        self.assertAlmostEqual(0.1 + 0.2, 0.3)
        self.assertAlmostEqual(1.0, 1.05, delta=0.1)
        self.assertLess(1, 2)
        self.assertGreater(2.5, 1.5)

    def test_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            parse("x")
        self.assertEqual(str(cm.exception), "invalid literal for int() with base 10: 'x'")
        self.assertRaisesCall(ZeroDivisionError, divide, 1, 0)

    def test_isinstance(self) -> None:
        self.assertIsInstance(self.values, list)


class FixtureTests(unittest.TestCase):
    log: list[str] = []

    @classmethod
    def setUpClass(cls) -> None:
        FixtureTests.log.append("setUpClass")

    @classmethod
    def tearDownClass(cls) -> None:
        FixtureTests.log.append("tearDownClass")
        print("fixture log:", FixtureTests.log)

    def note(self, text: str) -> None:
        FixtureTests.log.append(text)

    def setUp(self) -> None:
        FixtureTests.log.append("setUp " + self._testMethodName)
        # WORKAROUND(ut_unsupported_bound_builtin_method_as_callable): a builtin's bound method (`log.append`) is not a callable value.
        self.addCleanup(self.note, "cleanup " + self._testMethodName)

    def tearDown(self) -> None:
        FixtureTests.log.append("tearDown " + self._testMethodName)

    def test_a(self) -> None:
        FixtureTests.log.append("a")

    def test_b(self) -> None:
        FixtureTests.log.append("b")


class DecoratedTests(unittest.TestCase):
    @unittest.skip("not today")
    def test_skipped(self) -> None:
        self.fail("should not run")

    @unittest.skipIf(True, "condition held")
    def test_skipped_if(self) -> None:
        self.fail("should not run")

    @unittest.skipUnless(True, "condition held")
    def test_not_skipped(self) -> None:
        self.assertTrue(True)

    @unittest.expectedFailure
    def test_expected_failure(self) -> None:
        self.assertEqual(1, 2)

    def test_skip_inside(self) -> None:
        self.skipTest("skipped from the body")

    def test_subtests(self) -> None:
        for i in range(3):
            self.assertLess(i, 3)


unittest.main([ArithmeticTests, FixtureTests, DecoratedTests])
