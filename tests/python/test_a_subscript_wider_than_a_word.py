# An int too wide for a 64-bit word used as a sequence subscript -- read,
# assigned, deleted, on a list, tuple, str, bytes, bytearray and memoryview,
# and held as an int the program computed -- raises IndexError, as CPython's
# PyNumber_AsSsize_t(v, IndexError) does; as an index argument of a method
# (`list.pop`, `list.insert`) it raises OverflowError.
import unittest

WIDE = 2 ** 70


class SubscriptTest(unittest.TestCase):

    def test_reads(self) -> None:
        xs = [1, 2, 3]
        with self.assertRaises(IndexError) as cm:
            xs[WIDE]
        self.assertEqual(str(cm.exception),
                         "cannot fit 'int' into an index-sized integer")
        with self.assertRaises(IndexError):
            xs[-WIDE]
        with self.assertRaises(IndexError):
            (1, 2)[WIDE]
        with self.assertRaises(IndexError):
            "ab"[WIDE]
        with self.assertRaises(IndexError):
            b"ab"[WIDE]
        with self.assertRaises(IndexError):
            memoryview(b"ab")[WIDE]

    def test_writes_and_deletes(self) -> None:
        xs = [1, 2, 3]
        with self.assertRaises(IndexError):
            xs[WIDE] = 5
        with self.assertRaises(IndexError):
            del xs[WIDE]
        ba = bytearray(b"ab")
        with self.assertRaises(IndexError):
            ba[WIDE] = 1
        self.assertEqual((xs, ba), ([1, 2, 3], bytearray(b"ab")))

    def test_a_computed_index(self) -> None:
        xs = [1, 2, 3]
        i = WIDE
        with self.assertRaises(IndexError):
            xs[i - 1]
        self.assertEqual(xs[i - WIDE + 1], 2)

    def test_a_method_argument(self) -> None:
        xs = [1, 2, 3]
        with self.assertRaises(OverflowError):
            xs.pop(WIDE)
        with self.assertRaises(OverflowError):
            xs.insert(WIDE, 9)


if __name__ == "__main__":
    unittest.main()
