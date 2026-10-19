# WHAT: `range.count` is declared by the typeshed contract but has no
#   implementation.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   builtins.range.count is declared by the standard-library contract but has no implementation in Lython, so the call cannot be lowered; the declaration is the defect, not the call
# CPython 3.14 expects: 1
print(range(10).count(3))
