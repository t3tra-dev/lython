# WHAT: An instance assignment over a class attribute that holds a class
#   value. For a `str` attribute the emitter refuses with a reason (see
#   ut_unsupported_instance_shadows_class_attribute); for `type[...]` it
#   lets it through and the lowering stops with "class Case has no field
#   'failureException'". unittest's _SubTest copies its parent's
#   failureException this way.
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- an instance store makes the class attribute's line carry an
#   unset field; a read takes the field when set, type(obj).attr otherwise.
# CLASSIFICATION @ lang-gaps 51e1e5ed + the fix: 1 正しい
# CPython 3.14 expects: AssertionError ValueError
class Case:
    failureException: type[BaseException] = AssertionError


a = Case()
b = Case()
b.failureException = ValueError
print(a.failureException.__name__, b.failureException.__name__)
