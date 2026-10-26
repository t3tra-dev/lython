# assertRaises and assertRaisesRegex take a tuple of exception classes as well
# as one: the block passes when it raises any of them, the context's
# exception is the one raised, another exception passes through, and a block
# that raises nothing fails naming the tuple.
import unittest


class TupleTest(unittest.TestCase):

    def test_any_of_them(self) -> None:
        with self.assertRaises((KeyError, IndexError)):
            [1][5]
        with self.assertRaises((KeyError, IndexError)) as cm:
            {1: 2}[3]
        self.assertEqual(type(cm.exception).__name__, "KeyError")
        with self.assertRaisesRegex((ValueError, KeyError), "boom"):
            raise ValueError("boom here")

    def test_one_class_still(self) -> None:
        with self.assertRaises(KeyError) as cm:
            {}[1]
        self.assertEqual(str(cm.exception), "1")

    def test_nothing_raised(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            with self.assertRaises((KeyError, IndexError)):
                pass
        self.assertEqual(str(cm.exception),
                         "(<class 'KeyError'>, <class 'IndexError'>) not raised")

    def test_another_exception_passes_through(self) -> None:
        with self.assertRaises(ValueError):
            with self.assertRaises((KeyError, IndexError)):
                raise ValueError("v")


if __name__ == "__main__":
    unittest.main()
