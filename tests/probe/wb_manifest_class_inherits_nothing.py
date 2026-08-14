# FIXED 2026-08-15 for the manifest index; the residue below is a DIFFERENT
# defect that the first draft of this file misattributed to inheritance.
#
# WHAT WAS WRONG: `py.class @bool` declares nine methods and
# `base_names = ["int"]` (runtime/modules/builtins.mlir:209), and CPython's bool
# inherits every one of int's. Three tables answered differently about that:
#
#   1. the protocol/contract table -- KNEW the base. `ProtocolInfo::bases` is
#      populated from `base_names` (PyProtocols.cpp:884), which is why
#      True.__int__() passed the emitter.
#   2. the manifest implementation index -- DID NOT. `methodCandidates(
#      "builtins.bool", "__abs__")` was empty and its only rescues were named
#      special cases: exception ancestors, and the dunders every class inherits
#      from object. So abs(True), True.__int__(), __index__, __invert__ and
#      round(True, 0) all passed the emitter and died in lowering with
#      "declared by the standard-library contract but has no implementation".
#   3. the emitter's manifest lookup -- went through table 1, so it agreed.
#
# `selectManifestMethod` now walks the base chain, reading it from the protocol
# table because `classForContract` does not find manifest classes. Pinned by
# tests/golden/cases/bool_inherits_int_methods.py; the receiver widening those
# calls need was landed the same day in `appendRuntimeSource`.
#
# ⛔ Two claims in the first draft of this file were WRONG and are recorded so
# the measurements are not repeated:
#
#   `b.bit_length()` is NOT evidence of the inheritance gap. int does not
#   declare bit_length at all -- `n: int = 5; n.bit_length()` is refused too.
#   That is a missing method on int, and it has nothing to do with bool.
#
#   `round(True)` is NOT the inheritance gap either. int's `__round__` contract
#   declares ndigits as a REQUIRED second parameter, so the no-argument form is
#   refused for a plain int as well:
#
#     n: int = 5
#     n.__round__()  ... refused    n.__round__(0) ... 5
#     round(n) ....... 5            round(True) .... refused
#
#   round(n) works because the `round` builtin carries its own one-argument
#   contract; the dunder does not. CPython's int.__round__ takes ndigits
#   optionally, and the default exists here only as `ly.runtime.default_i64` on
#   LyLong_Round (builtins.mlir:8122) -- a runtime fact the protocol contract
#   does not carry. THAT is the remaining defect, and it is about optional
#   parameters in manifest contracts, not about bases.
#
# differential: skip refused; the point is the refusal

n: int = 5
print(n.__round__())
