"""Unittest main program"""

import sys
import os

from . import loader, runner, suite

__unittest = True
_NO_TESTS_EXITCODE = 5


def _convert_select_pattern(pattern: str) -> str:
    if not '*' in pattern:
        pattern = '*%s*' % pattern
    return pattern


class TestProgram(object):
    """A command-line program that runs a set of tests; this is primarily
       for making test modules conveniently executable.
    """
    # defaults for testing
    module: str | None = None
    verbosity = 1
    failfast: bool | None = None
    catchbreak: bool | None = None
    buffer: bool | None = None
    progName: str | None = None
    warnings: str | None = None
    testNamePatterns: list[str] | None = None

    def __init__(self, module: str | None = '__main__', defaultTest: str | None = None,
                 argv: list[str] | None = None,
                 testRunner: type[runner.TextTestRunner] | None = None,
                 testLoader: loader.TestLoader = loader.defaultTestLoader,
                 exit: bool = True, verbosity: int = 1, failfast: bool | None = None,
                 catchbreak: bool | None = None, buffer: bool | None = None,
                 warnings: str | None = None, *, tb_locals: bool = False,
                 durations: int | None = None) -> None:
        self.module = module
        if argv is None:
            argv = sys.argv

        self.exit = exit
        self.failfast = failfast
        self.catchbreak = catchbreak
        self.verbosity = verbosity
        self.buffer = buffer
        self.tb_locals = tb_locals
        self.durations = durations
        self.warnings = warnings
        self.defaultTest = defaultTest
        self.testRunner = testRunner
        self.testLoader = testLoader
        self.progName = os.path.basename(argv[0])
        self.parseArgs(argv)
        self.runTests()

    def parseArgs(self, argv: list[str]) -> None:
        args = argv[1:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == '-v' or arg == '--verbose':
                self.verbosity = 2
            elif arg == '-q' or arg == '--quiet':
                self.verbosity = 0
            elif arg == '-f' or arg == '--failfast':
                self.failfast = True
            elif arg == '-b' or arg == '--buffer':
                self.buffer = True
            elif arg == '-k' and i + 1 < len(args):
                i += 1
                patterns = self.testNamePatterns
                if patterns is None:
                    patterns = []
                    self.testNamePatterns = patterns
                patterns.append(_convert_select_pattern(args[i]))
            elif arg.startswith('--durations='):
                self.durations = int(arg[len('--durations='):])
            i += 1
        self.createTests()

    def createTests(self) -> None:
        if self.testNamePatterns:
            self.testLoader.testNamePatterns = self.testNamePatterns
        module = self.module
        if module is None:
            self.test = suite.TestSuite()
        else:
            self.test = self.testLoader.loadTestsFromModule(module)

    def runTests(self) -> None:
        testRunnerClass = self.testRunner
        if testRunnerClass is None:
            testRunnerClass = runner.TextTestRunner
        testRunner = testRunnerClass(verbosity=self.verbosity,
                                     failfast=bool(self.failfast),
                                     buffer=bool(self.buffer),
                                     warnings=self.warnings,
                                     tb_locals=self.tb_locals,
                                     durations=self.durations)
        self.result = testRunner.run(self.test)
        if self.exit:
            if not self.result.wasSuccessful():
                sys.exit(1)
            elif self.result.testsRun == 0 and len(self.result.skipped) == 0:
                sys.exit(_NO_TESTS_EXITCODE)
            else:
                sys.exit(0)

main = TestProgram
