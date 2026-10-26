# An assertRaises context's `expected` is what the call was given -- one
# class, or the tuple of classes -- and its failure message names it the way
# CPython's does.
import unittest


class AssertRaisesExpectedTest(unittest.TestCase):

    def test_one_class(self) -> None:
        with self.assertRaises(KeyError) as cm:
            {}[1]
        self.assertIs(cm.expected, KeyError)

    def test_a_tuple_of_classes(self) -> None:
        with self.assertRaises((KeyError, IndexError)) as cm:
            [][1]
        expected = cm.expected
        assert isinstance(expected, tuple)
        self.assertEqual(expected, (KeyError, IndexError))
        self.assertIsInstance(cm.exception, IndexError)

    def test_not_raised_names_what_was_expected(self) -> None:
        with self.assertRaises(AssertionError) as one:
            with self.assertRaises(KeyError):
                pass
        self.assertEqual(str(one.exception), "KeyError not raised")
        with self.assertRaises(AssertionError) as both:
            with self.assertRaises((KeyError, IndexError)):
                pass
        self.assertEqual(str(both.exception),
                         "(<class 'KeyError'>, <class 'IndexError'>) not raised")


if __name__ == "__main__":
    unittest.main()
