# WHAT: the framework running other test classes -- what a TestResult records
# for each outcome, the order of fixtures and cleanups, subtests, docstrings in
# descriptions, ids, a failing setUpClass, and the text a runner writes.
import io
import unittest

import unittest_samples as samples


def run(case: type[unittest.TestCase]) -> unittest.TestResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(case)
    result = unittest.TestResult()
    suite.run(result)
    return result


class ResultRecords(unittest.TestCase):
    def test_counts(self) -> None:
        result = run(samples.Outcomes)
        self.assertEqual(result.testsRun, 7)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(len(result.skipped), 2)
        self.assertEqual(len(result.expectedFailures), 1)
        self.assertEqual(len(result.unexpectedSuccesses), 1)
        self.assertFalse(result.wasSuccessful())

    def test_which_test_each_record_names(self) -> None:
        result = run(samples.Outcomes)
        self.assertEqual(result.failures[0][0].id(),
                         "unittest_samples.Outcomes.test_b_fails")
        self.assertEqual(result.errors[0][0].id(),
                         "unittest_samples.Outcomes.test_c_errors")
        self.assertEqual([reason for _, reason in result.skipped],
                         ["not now", "inside"])

    def test_tracebacks_end_with_the_exception(self) -> None:
        result = run(samples.Outcomes)
        self.assertTrue(result.failures[0][1].endswith("AssertionError: 1 != 2\n"))
        self.assertTrue(result.errors[0][1].endswith("KeyError: 'k'\n"))
        self.assertIn("Traceback (most recent call last):", result.errors[0][1].split("\n"))


class FixtureOrder(unittest.TestCase):
    def test_order(self) -> None:
        samples.log.clear()
        result = run(samples.Fixtures)
        self.assertEqual(samples.log, [
            "setUpClass",
            "setUp test_one", "one", "tearDown test_one", "cleanup test_one",
            "setUp test_two", "two", "tearDown test_two", "cleanup test_two",
            "tearDownClass",
        ])
        self.assertEqual(len(result.failures), 1)

    def test_a_failing_class_setup_runs_no_test(self) -> None:
        samples.log.clear()
        result = run(samples.BrokenClassSetUp)
        self.assertEqual(samples.log, [])
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(str(result.errors[0][0]),
                         "setUpClass (unittest_samples.BrokenClassSetUp)")


class SubTestRecords(unittest.TestCase):
    def test_each_failing_subtest_is_a_failure(self) -> None:
        result = run(samples.SubTests)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual([str(test) for test, _ in result.failures], [
            "test_numbers (unittest_samples.SubTests.test_numbers) (n=1)",
            "test_numbers (unittest_samples.SubTests.test_numbers) (n=3)",
        ])


class Descriptions(unittest.TestCase):
    def test_short_description_is_the_first_doc_line(self) -> None:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(samples.Described)
        tests = [t for t in suite if isinstance(t, unittest.TestCase)]
        self.assertEqual([t.shortDescription() for t in tests],
                         ["First line of the doc.", None])
        self.assertEqual(str(tests[1]),
                         "test_plain (unittest_samples.Described.test_plain)")
        self.assertEqual(repr(tests[1]),
                         "<unittest_samples.Described testMethod=test_plain>")


class RunnerText(unittest.TestCase):
    def test_verbose_report(self) -> None:
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(samples.Described))
        lines = stream.getvalue().split("\n")
        self.assertEqual(lines[:3], [
            "test_documented (unittest_samples.Described.test_documented)",
            "First line of the doc. ... ok",
            "test_plain (unittest_samples.Described.test_plain) ... ok",
        ])
        self.assertEqual(lines[-2], "OK")

    def test_dots_and_summary(self) -> None:
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream)
        result = runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(samples.Outcomes))
        lines = stream.getvalue().split("\n")
        self.assertEqual(lines[0], ".FEsxus")
        self.assertEqual(lines[-2],
                         "FAILED (failures=1, errors=1, skipped=2, expected failures=1, "
                         "unexpected successes=1)")
        self.assertFalse(result.wasSuccessful())


if __name__ == "__main__":
    unittest.main()
