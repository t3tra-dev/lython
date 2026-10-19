# WHAT: An instance attribute assigned over a class attribute of the same
#   name: "would shadow the class attribute; initialize it in __init__".
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- see ut_bug_class_value_class_attribute_shadowed.
# CLASSIFICATION @ lang-gaps 51e1e5ed + the fix: 1 正しい
# CPython 3.14 expects: case b
class Case:
    label: str = "case"


a = Case()
b = Case()
b.label = "b"
print(a.label, b.label)
