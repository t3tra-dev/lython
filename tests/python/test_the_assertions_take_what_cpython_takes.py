# The assertion methods take what CPython's take: the two sides of a
# comparison need not have one type (an int list equals a float list), a
# membership test takes any container, an order test any two comparable values,
# and an instance or subclass test a tuple of classes. The methods CPython 3.14
# has beside them -- assertLessEqual, assertGreaterEqual, assertNotAlmostEqual,
# assertCountEqual, assertNotIsInstance, assertIsSubclass, assertNotIsSubclass
# and the four affix tests -- pass and fail as CPython's do, with CPython's
# messages.
import unittest


class AssertionSurfaceTest(unittest.TestCase):
    def test_equal_across_numeric_types(self):
        self.assertEqual([1, 2], [1.0, 2.0])
        self.assertEqual((1, 2), (1.0, 2.0))
        self.assertEqual({"a": 1}, {"a": 1.0})
        self.assertEqual(1, 1.0)
        self.assertNotEqual([1], [2.0])
        self.assertNotEqual("a", 1)

    def test_in_any_container(self):
        self.assertIn("a", "abc")
        self.assertNotIn("z", "abc")
        self.assertIn(1, (1, 2))
        self.assertIn("k", {"k": 1})
        self.assertIn(1, {1, 2})
        self.assertIn(3, range(5))
        self.assertIn(1, [1, 2])

    def test_order(self):
        self.assertLess("a", "b")
        self.assertLess(1, 1.5)
        self.assertLessEqual(1, 1)
        self.assertGreater((2, 1), (1, 9))
        self.assertGreaterEqual(2.5, 2)

    def test_almost(self):
        self.assertAlmostEqual(1.0, 1.00000001)
        self.assertNotAlmostEqual(1.0, 1.1)
        self.assertNotAlmostEqual(1.0, 1.5, delta=0.25)

    def test_count_equal(self):
        self.assertCountEqual([1, 2, 2], [2, 1, 2])
        self.assertCountEqual("abc", "cab")
        self.assertCountEqual([[1], [2]], [[2], [1]])

    def test_instance_and_subclass(self):
        self.assertIsInstance(1, (int, str))
        self.assertNotIsInstance(1, str)
        self.assertNotIsInstance(1.5, (int, str))
        self.assertIsSubclass(bool, int)
        self.assertIsSubclass(bool, (str, int))
        self.assertNotIsSubclass(int, str)

    def test_affixes(self):
        self.assertStartsWith("abc", "ab")
        self.assertStartsWith("abc", ("x", "a"))
        self.assertNotStartsWith("abc", "x")
        self.assertEndsWith("abc", "bc")
        self.assertNotEndsWith("abc", ("x", "y"))

    def test_membership_and_order_messages(self):
        with self.assertRaises(AssertionError) as cm:
            self.assertIn("z", "abc")
        self.assertEqual(str(cm.exception), "'z' not found in 'abc'")
        with self.assertRaises(AssertionError) as cm:
            self.assertNotIn("k", {"k": 1})
        self.assertEqual(str(cm.exception),
                         "'k' unexpectedly found in {'k': 1}")
        with self.assertRaises(AssertionError) as cm:
            self.assertLessEqual(2, 1)
        self.assertEqual(str(cm.exception), "2 not less than or equal to 1")
        with self.assertRaises(AssertionError) as cm:
            self.assertGreaterEqual(1, 2.5)
        self.assertEqual(str(cm.exception),
                         "1 not greater than or equal to 2.5")
        with self.assertRaises(AssertionError) as cm:
            self.assertLess("b", "a")
        self.assertEqual(str(cm.exception), "'b' not less than 'a'")

    def test_almost_and_count_messages(self):
        with self.assertRaises(AssertionError) as cm:
            self.assertNotAlmostEqual(1.0, 1.0)
        self.assertEqual(str(cm.exception), "1.0 == 1.0 within 7 places")
        with self.assertRaises(AssertionError) as cm:
            self.assertNotAlmostEqual(1.0, 1.1, delta=0.5)
        self.assertEqual(
            str(cm.exception),
            "1.0 == 1.1 within 0.5 delta (0.10000000000000009 difference)")
        with self.assertRaises(AssertionError) as cm:
            self.assertCountEqual([1, 1, 2], [1, 2, 3])
        self.assertEqual(str(cm.exception),
                         "Element counts were not equal:\n"
                         "First has 2, Second has 1:  1\n"
                         "First has 0, Second has 1:  3")
        with self.assertRaises(AssertionError) as cm:
            self.assertCountEqual([[1], [1]], [[1], [2]])
        self.assertEqual(str(cm.exception),
                         "Element counts were not equal:\n"
                         "First has 2, Second has 1:  [1]\n"
                         "First has 0, Second has 1:  [2]")

    def test_class_and_affix_messages(self):
        with self.assertRaises(AssertionError) as cm:
            self.assertNotIsInstance(1, (str, int))
        self.assertEqual(str(cm.exception), "1 is an instance of <class 'int'>")
        with self.assertRaises(AssertionError) as cm:
            self.assertIsInstance(1.5, (int, str))
        self.assertEqual(
            str(cm.exception),
            "1.5 is not an instance of any of (<class 'int'>, <class 'str'>)")
        with self.assertRaises(AssertionError) as cm:
            self.assertIsSubclass(int, str)
        self.assertEqual(str(cm.exception),
                         "<class 'int'> is not a subclass of <class 'str'>")
        with self.assertRaises(AssertionError) as cm:
            self.assertNotIsSubclass(bool, (str, int))
        self.assertEqual(str(cm.exception),
                         "<class 'bool'> is a subclass of <class 'int'>")
        with self.assertRaises(AssertionError) as cm:
            self.assertStartsWith("abc", ("x", "y"))
        self.assertEqual(str(cm.exception),
                         "'abc' doesn't start with any of ('x', 'y')")
        with self.assertRaises(AssertionError) as cm:
            self.assertNotEndsWith("abc", ("x", "c"))
        self.assertEqual(str(cm.exception), "'abc' ends with 'c'")

    def test_a_list_difference_across_numeric_types(self):
        with self.assertRaises(AssertionError) as cm:
            self.assertEqual([1, 2], [1.0, 3.0])
        self.assertEqual(str(cm.exception),
                         "Lists differ: [1, 2] != [1.0, 3.0]\n\n"
                         "First differing element 1:\n2\n3.0\n\n"
                         "- [1, 2]\n+ [1.0, 3.0]")


if __name__ == "__main__":
    unittest.main()
