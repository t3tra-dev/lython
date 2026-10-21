# Test classes that tests/python/test_unittest_runner.py runs through the
# framework; not a test module itself (no `test_` prefix), and imported as a
# module so that `unittest.main()` does not collect these classes as its own.
import unittest

log: list[str] = []


class Outcomes(unittest.TestCase):
    def test_a_passes(self) -> None:
        pass

    def test_b_fails(self) -> None:
        self.assertEqual(1, 2)

    def test_c_errors(self) -> None:
        raise KeyError("k")

    @unittest.skip("not now")
    def test_d_skipped(self) -> None:
        pass

    @unittest.expectedFailure
    def test_e_expected(self) -> None:
        self.assertEqual(1, 2)

    @unittest.expectedFailure
    def test_f_unexpected(self) -> None:
        pass

    def test_g_skip_inside(self) -> None:
        self.skipTest("inside")


class Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        log.append("setUpClass")

    @classmethod
    def tearDownClass(cls) -> None:
        log.append("tearDownClass")

    def setUp(self) -> None:
        log.append("setUp " + self._testMethodName)
        self.addCleanup(log.append, "cleanup " + self._testMethodName)

    def tearDown(self) -> None:
        log.append("tearDown " + self._testMethodName)

    def test_one(self) -> None:
        log.append("one")

    def test_two(self) -> None:
        log.append("two")
        self.fail("two fails")


class SubTests(unittest.TestCase):
    def test_numbers(self) -> None:
        for n in range(4):
            with self.subTest(n=n):
                self.assertNotEqual(n % 2, 1)


class Described(unittest.TestCase):
    def test_documented(self) -> None:
        """First line of the doc.

        More text.
        """

    def test_plain(self) -> None:
        pass


class BrokenClassSetUp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raise ValueError("no setup")

    def test_never_runs(self) -> None:
        log.append("never")
