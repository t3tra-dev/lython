# WHAT: skip, skipIf and skipUnless on a TestCase class -- every test of the
# class and of its subclasses is skipped with the class's reason and its
# setUpClass / tearDownClass do not run -- beside a class whose condition
# lets it run, compared with what CPython's unittest records.
import unittest

calls: list[str] = []


@unittest.skip("not today")
class Skipped(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        calls.append("setUpClass")

    @classmethod
    def tearDownClass(cls) -> None:
        calls.append("tearDownClass")

    def test_a(self) -> None:
        calls.append("test_a")


class Child(Skipped):
    def test_b(self) -> None:
        calls.append("test_b")


@unittest.skipUnless(False, "needs a feature")
class Unless(unittest.TestCase):
    def test_c(self) -> None:
        calls.append("test_c")


@unittest.skipIf(False, "never")
class Runs(unittest.TestCase):
    def test_d(self) -> None:
        calls.append("test_d")


class ClassSkip(unittest.TestCase):
    def run_class(self, cls: type[unittest.TestCase]) -> unittest.TestResult:
        calls.clear()
        result = unittest.TestResult()
        unittest.TestLoader().loadTestsFromTestCase(cls).run(result)
        return result

    def test_skipped_class(self) -> None:
        result = self.run_class(Skipped)
        self.assertEqual([(t.id().split('.')[-1], why) for t, why in result.skipped],
                         [('test_a', 'not today')])
        self.assertEqual(calls, [])

    def test_subclass_inherits_the_skip(self) -> None:
        result = self.run_class(Child)
        self.assertEqual(sorted(why for t, why in result.skipped),
                         ['not today', 'not today'])
        self.assertEqual(calls, [])

    def test_skip_unless_and_skip_if(self) -> None:
        unless = self.run_class(Unless)
        self.assertEqual([why for t, why in unless.skipped], ['needs a feature'])
        runs = self.run_class(Runs)
        self.assertEqual(runs.skipped, [])
        self.assertEqual(runs.testsRun, 1)
        self.assertEqual(calls, ['test_d'])


if __name__ == '__main__':
    unittest.main()
