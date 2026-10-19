# WHAT: nine tests -- six fail, one errors, one passes against its
# expectedFailure, one passes -- run by the prototype unittest next to this
# file. Under python3.14 and under `lyc jit` the progress line, the
# FAIL/ERROR/UNEXPECTED SUCCESS headings, the summary and the exit status are
# the same. The reports differ only in what
# ut_bug_contextmanager_handler_traceback_lacks_body_frames records: under lyc
# no frame of the test is in the traceback, so only the exception line is
# printed.
import unittest


def parse(text: str) -> int:
    return int(text)


class FailingTests(unittest.TestCase):
    def test_equal_ints(self) -> None:
        self.assertEqual(1 + 1, 3)

    def test_equal_strings(self) -> None:
        self.assertEqual("abc", "abd", "with a message")

    def test_lists(self) -> None:
        self.assertListEqual([1, 2, 3], [1, 2, 4, 5])

    def test_error(self) -> None:
        parse("not a number")

    def test_not_raised(self) -> None:
        with self.assertRaises(KeyError):
            parse("1")

    def test_true(self) -> None:
        self.assertTrue(len("") > 0)

    def test_subtests(self) -> None:
        for i in range(3):
            with self.subTest(i=i):
                self.assertLess(i, 2)
        with self.subTest("outer", n=1):
            with self.subTest(None, m=2):
                self.assertEqual("a", "b")

    @unittest.expectedFailure
    def test_unexpected_success(self) -> None:
        self.assertEqual(1, 1)

    def test_passes(self) -> None:
        self.assertIn(3, [1, 2, 3])


unittest.main(FailingTests)
