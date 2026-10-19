# WHAT: A constant an IMPORTED module computes from other constants (`_DIFF
#   = _MAX - (_PLACE + 5)`) needs an annotation there: "a scalar travels as
#   a literal". unittest/util.py has five of these.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   '_DIFF' is assigned at the top level of this imported module, but its type is not one a module global can hold: a scalar travels as a literal and a container gets a cell only when its element type is resolved. Annotate i
# CPython 3.14 expects: 63
import ut_mod_computed_constant

print(ut_mod_computed_constant.show())
