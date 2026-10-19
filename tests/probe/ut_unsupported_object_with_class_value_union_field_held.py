# WHAT: An object whose class has a `type[Case] | None` field cannot be held
#   by another object's field or a list: it "expands to 3 physical values
#   and nothing can rebuild them from one address". unittest's TestResult
#   has `_previousTestClass: type | None`.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   a '!py.contract<"Result">' value expands to 3 physical values and nothing can rebuild them from one address; it cannot be stored in a container slot or boxed field yet (a union with two `type[X]` members keeps every memb
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
