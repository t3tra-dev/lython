# WHAT: A class attribute holding a class value (`failureException:
#   type[BaseException] = AssertionError`) read through an instance in a
#   function. The emitter accepts it; the lowering stops with "class Case
#   has no field 'failureException'". (With a subclass overriding it the
#   emitter refuses, see
#   ut_unsupported_class_attribute_overridden_read_through_base.)
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- a class attribute holding a class value is slot-backed like
#   a module global holding one.
# CLASSIFICATION @ lang-gaps 51e1e5ed + the fix: 1 正しい
# CPython 3.14 expects: True False
class Case:
    failureException: type[BaseException] = AssertionError


def is_failure(t: type[BaseException], test: Case) -> bool:
    return issubclass(t, test.failureException)


print(is_failure(AssertionError, Case()), is_failure(KeyError, Case()))
