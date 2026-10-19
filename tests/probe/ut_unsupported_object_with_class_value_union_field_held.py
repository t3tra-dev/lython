# WHAT: An object whose class has a `type[Case] | None` field cannot be held
#   by another object's field or a list: it "expands to 3 physical values
#   and nothing can rebuild them from one address". unittest's TestResult
#   has `_previousTestClass: type | None`.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- a `type[X] | None` field is boxed, by the general union rule.
# CLASSIFICATION @ lang-gaps 51e1e5ed + the fix: 1 正しい
# CPython 3.14 expects: True
class Case:
    pass


class Result:
    def __init__(self) -> None:
        self.n = 0
        self.previous: type[Case] | None = None


class Outcome:
    def __init__(self, result: Result) -> None:
        self.result = result


o = Outcome(Result())
o.result.previous = Case
print(o.result.previous is Case)
