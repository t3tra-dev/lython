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
#   declared ndigits as a REQUIRED second parameter, so the no-argument form was
#   refused for a plain int as well -- `n.__round__(0)` and `round(n)` both
#   worked, the latter through the `round` builtin's own one-argument contract.
#   A manifest method contract cannot spell an optional parameter; float already
#   declares __round__ twice, once per arity, and int now does the same. Fixed
#   separately, pinned by tests/golden/cases/round_without_ndigits.py.
#
# ⭐ WHAT THE TWO MISREADINGS HAVE IN COMMON: both took a symptom shared by
# several programs -- "bool cannot reach int's numeric methods" -- for a cause.
# Four causes at four depths produced it. What localised each one in a single
# measurement was running the SAME spelling on the BASE type: `n: int = 5;
# n.__round__()` and `n.bit_length()` both fail, which puts the defect above the
# derived class rather than in the inheritance between them.
#
# differential: run agrees with CPython now

n: int = 5
print(n.__round__(), round(n), round(True), n.__index__())
