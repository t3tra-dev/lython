# WHAT: a test module run by the standard `unittest` package -- `main` finding
# this module's TestCase subclasses and running them through a TextTestRunner
# subclass that writes to a StringIO -- with a pass, a failure whose two reprs
# are long enough to be shortened around their common prefix, an error, a
# skip, an expected failure, a subtest that fails, and class fixtures; the
# report is printed with its timing and file paths removed.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: unittest is a package of
# eight modules, two of which import each other, compiled from the sources
# lyc embeds; what it reports for each outcome is what running it says.
import io
import unittest


class Big:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Big) and other.tag == self.tag

    def __repr__(self) -> str:
        return "Big(" + "x" * 90 + self.tag + ")"


class Fixtures(unittest.TestCase):
    log: list[str] = []

    @classmethod
    def setUpClass(cls) -> None:
        Fixtures.log.append("setUpClass")

    @classmethod
    def tearDownClass(cls) -> None:
        Fixtures.log.append("tearDownClass")

    def test_pass(self) -> None:
        self.assertEqual(1 + 1, 2)

    def test_long_reprs(self) -> None:
        self.assertEqual(Big("abc"), Big("abd"))

    def test_error(self) -> None:
        raise KeyError("missing")

    @unittest.skip("not today")
    def test_skipped(self) -> None:
        pass

    @unittest.expectedFailure
    def test_known(self) -> None:
        self.assertEqual(1, 2)

    def test_sub(self) -> None:
        for n in range(3):
            with self.subTest(n=n):
                self.assertNotEqual(n, 1)


report = io.StringIO()


class BufferedRunner(unittest.TextTestRunner):
    def __init__(self, stream: io.TextIOWrapper | io.StringIO | None = None,
                 descriptions: bool = True, verbosity: int = 1,
                 failfast: bool = False, buffer: bool = False,
                 resultclass: type[unittest.TextTestResult] | None = None,
                 warnings: str | None = None, *, tb_locals: bool = False,
                 durations: int | None = None) -> None:
        super().__init__(report, descriptions, verbosity, failfast, buffer,
                         resultclass, warnings, tb_locals=tb_locals,
                         durations=durations)


program = unittest.main(module="__main__", argv=["tests"],
                        testRunner=BufferedRunner, exit=False, verbosity=2)
result = program.result
for line in report.getvalue().split("\n"):
    if line.startswith("Ran "):
        line = line[:line.index(" in ")]
    if line.startswith('  File "'):
        line = '  File "...", ' + line[line.index("line "):]
    print(line)
print(result.testsRun, len(result.failures), len(result.errors),
      len(result.skipped), len(result.expectedFailures), result.wasSuccessful())
print(Fixtures.log)
