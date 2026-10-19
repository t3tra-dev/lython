# WHAT: An instance attribute assigned over a class attribute of the same
#   name: "would shadow the class attribute; initialize it in __init__".
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   assigning 'label' through an instance would shadow the class attribute; initialize it in __init__ to make it an instance field
# CPython 3.14 expects: case b
class Case:
    label: str = "case"


a = Case()
b = Case()
b.label = "b"
print(a.label, b.label)
