# WHAT: A class attribute a subclass overrides (`failureException =
#   KeyError`) read through a base-typed receiver: "is overridden by a
#   subclass ... cannot be resolved from the static type of the receiver".
#   unittest reads failureException, maxDiff and longMessage this way.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   'failureException' is overridden by a subclass of 'Case', so this call cannot be resolved from the static type of the receiver
# CPython 3.14 expects: AssertionError KeyError
class Case:
    failureException: type[BaseException] = AssertionError


class Mine(Case):
    failureException: type[BaseException] = KeyError


def name_of(test: Case) -> str:
    return test.failureException.__name__


print(name_of(Case()), name_of(Mine()))
