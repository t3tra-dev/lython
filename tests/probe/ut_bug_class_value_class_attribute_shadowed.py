# WHAT: An instance assignment over a class attribute that holds a class
#   value. For a `str` attribute the emitter refuses with a reason (see
#   ut_unsupported_instance_shadows_class_attribute); for `type[...]` it
#   lets it through and the lowering stops with "class Case has no field
#   'failureException'". unittest's _SubTest copies its parent's
#   failureException this way.
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   class Case has no field 'failureException'
# CPython 3.14 expects: AssertionError ValueError
class Case:
    failureException: type[BaseException] = AssertionError


a = Case()
b = Case()
b.failureException = ValueError
print(a.failureException.__name__, b.failureException.__name__)
