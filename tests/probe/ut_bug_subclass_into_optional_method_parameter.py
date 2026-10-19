# WHAT: A subclass instance passed to a METHOD parameter declared `Base |
#   None` is refused ("is declared Result | None and this call gives it
#   TextResult"); a free function with the same parameter accepts it.
#   unittest's `TestCase.run(result: TestResult | None = None)` receives a
#   TextTestResult.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- a value fits a union parameter when it fits a member.
# CLASSIFICATION @ lang-gaps e3a74bc1 + the fix: 1 正しい
# CPython 3.14 expects: 1 -1
class Result:
    def __init__(self) -> None:
        self.n = 0


class TextResult(Result):
    pass


class Case:
    def run(self, result: Result | None = None) -> int:
        if result is None:
            return -1
        return result.n + 1


print(Case().run(TextResult()), Case().run())
