# WHAT: A class attribute holding a class value (`failureException:
#   type[BaseException] = AssertionError`) read through an instance in a
#   function. The emitter accepts it; the lowering stops with "class Case
#   has no field 'failureException'". (With a subclass overriding it the
#   emitter refuses, see
#   ut_unsupported_class_attribute_overridden_read_through_base.)
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   class Case has no field 'failureException'
# CPython 3.14 expects: True False
class Case:
    failureException: type[BaseException] = AssertionError


def is_failure(t: type[BaseException], test: Case) -> bool:
    return issubclass(t, test.failureException)


print(is_failure(AssertionError, Case()), is_failure(KeyError, Case()))
