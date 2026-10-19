# WHAT: A class attribute a subclass overrides (`failureException =
#   KeyError`) read through a base-typed receiver: "is overridden by a
#   subclass ... cannot be resolved from the static type of the receiver".
#   unittest reads failureException, maxDiff and longMessage this way.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- the attribute dispatcher pins a result type it cannot spell.
# CLASSIFICATION @ lang-gaps 51e1e5ed + the fix: 1 正しい
# CPython 3.14 expects: AssertionError KeyError
class Case:
    failureException: type[BaseException] = AssertionError


class Mine(Case):
    failureException: type[BaseException] = KeyError


def name_of(test: Case) -> str:
    return test.failureException.__name__


print(name_of(Case()), name_of(Mine()))
