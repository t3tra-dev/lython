# WHAT: unittest's assertion methods -- each passing case, and the message each
# failing one raises, compared with the text CPython's unittest produces.
import unittest


class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Point) and other.x == self.x and other.y == self.y

    def __repr__(self) -> str:
        return "Point(%d, %d)" % (self.x, self.y)


class PassingAssertions(unittest.TestCase):
    def test_equality(self) -> None:
        self.assertEqual(2 + 2, 4)
        self.assertEqual("ab" + "c", "abc")
        self.assertEqual(Point(1, 2), Point(1, 2))
        self.assertNotEqual(Point(1, 2), Point(2, 1))

    def test_truth_and_identity(self) -> None:
        self.assertTrue([1])
        self.assertFalse("")
        marker = object()
        self.assertIs(marker, marker)
        self.assertIsNot(marker, object())
        self.assertIsNone(None)
        self.assertIsNotNone(0)

    def test_containers(self) -> None:
        self.assertIn(3, [1, 2, 3])
        self.assertNotIn(4, [1, 2, 3])
        self.assertListEqual([1, 2], [1, 2])
        self.assertDictEqual({"a": 1}, {"a": 1})
        self.assertSetEqual({1, 2}, {2, 1})

    def test_numbers(self) -> None:
        self.assertAlmostEqual(0.1 + 0.2, 0.3)
        self.assertAlmostEqual(1.0, 1.04, delta=0.05)
        self.assertLess(1, 2)
        self.assertGreater(2.5, 1.0)
        self.assertIsInstance(True, int)

    def test_raises(self) -> None:
        with self.assertRaises(ZeroDivisionError):
            print(1 // 0)
        with self.assertRaises(KeyError) as cm:
            print({"a": 1}["b"])
        self.assertEqual(str(cm.exception), "'b'")


class FailureMessages(unittest.TestCase):
    def test_equal(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            self.assertEqual(1, 2)
        self.assertEqual(str(cm.exception), "1 != 2")

    def test_equal_with_msg(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            self.assertEqual(Point(1, 2), Point(3, 4), "points")
        self.assertEqual(str(cm.exception), "Point(1, 2) != Point(3, 4) : points")

    def test_long_reprs_are_shortened(self) -> None:
        long_a = Point(10 ** 40, 1)
        long_b = Point(10 ** 40, 2)
        with self.assertRaises(AssertionError) as cm:
            self.assertEqual(long_a, long_b)
        self.assertEqual(str(cm.exception), "%r != %r" % (long_a, long_b))

    def test_not_equal(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            self.assertNotEqual(5, 5)
        self.assertEqual(str(cm.exception), "5 == 5")

    def test_truth(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            self.assertTrue(0)
        self.assertEqual(str(cm.exception), "0 is not true")
        with self.assertRaises(AssertionError) as cm:
            self.assertFalse([1])
        self.assertEqual(str(cm.exception), "[1] is not false")

    def test_none(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            self.assertIsNone(3)
        self.assertEqual(str(cm.exception), "3 is not None")
        with self.assertRaises(AssertionError) as cm:
            self.assertIsNotNone(None)
        self.assertEqual(str(cm.exception), "unexpectedly None")

    def test_in(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            self.assertIn(4, [1, 2])
        self.assertEqual(str(cm.exception), "4 not found in [1, 2]")
        with self.assertRaises(AssertionError) as cm:
            self.assertNotIn(1, [1, 2])
        self.assertEqual(str(cm.exception), "1 unexpectedly found in [1, 2]")

    def test_ordering(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            self.assertLess(3, 2)
        self.assertEqual(str(cm.exception), "3 not less than 2")
        with self.assertRaises(AssertionError) as cm:
            self.assertGreater(1, 2)
        self.assertEqual(str(cm.exception), "1 not greater than 2")

    def test_almost_equal(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            self.assertAlmostEqual(1.0, 1.1)
        self.assertEqual(str(cm.exception),
                         "1.0 != 1.1 within 7 places (0.10000000000000009 difference)")

    def test_raises_not_raised(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            with self.assertRaises(ValueError):
                pass
        self.assertEqual(str(cm.exception), "ValueError not raised")

    def test_fail(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            self.fail("stop")
        self.assertEqual(str(cm.exception), "stop")


if __name__ == "__main__":
    unittest.main()
