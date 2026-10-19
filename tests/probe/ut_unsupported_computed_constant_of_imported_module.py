# WHAT: A constant an IMPORTED module computes from other constants (`_DIFF
#   = _MAX - (_PLACE + 5)`) needs an annotation there: "a scalar travels as
#   a literal". unittest/util.py has five of these.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- int and str arithmetic over the module's constants folds to
#   a literal, as the constants it is computed from travel.
# CLASSIFICATION @ lang-gaps e3a74bc1 + the fix: 1 正しい
# CPython 3.14 expects: 63
import ut_mod_computed_constant

print(ut_mod_computed_constant.show())
