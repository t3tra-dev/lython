# OPEN, and NEW (found 2026-08-15) -- the cause behind abs(True) and
# round(True), which tests/probe/wb_bool_argument_to_numeric_builtin.py had
# attributed to the numeric tower. It is not numeric at all:
#
#   b: bool = True
#   b.bit_length() .... bool does not provide manifest method 'bit_length'
#   b.__abs__() ....... the same
#   abs(True) ......... builtins.bool.__abs__ is declared by the standard-library
#                       contract but has no implementation in Lython
#   round(True) ....... bool does not provide manifest method '__round__'
#
# `py.class @bool` declares `base_names = ["int"]` and nine methods of its own
# (runtime/modules/builtins.mlir:209). CPython's bool inherits every one of
# int's; here it inherits NONE. bit_length is the proof that this is not about
# abs, round, or the numeric tower.
#
# ⭐ THREE TABLES DISAGREE ABOUT INHERITANCE, and each failure above is one of
# them answering:
#
#   1. the protocol/contract table -- KNOWS the base. It is what lets abs(True)
#      past the emitter: `methodContractCandidatesWithEvidence` finds int's
#      __abs__ declaration through bool, which is why the lowering diagnostic
#      can say "declared by the standard-library contract"
#      (Runtime/Manifest/Calls.cpp:74).
#   2. the manifest implementation index -- DOES NOT. `methodCandidates(
#      "builtins.bool", "__abs__")` is empty and Manifest/Index.h carries no
#      base information at all, so there is nothing to walk. Its two rescues are
#      named special cases: exception ancestors, and the dunders every class
#      inherits from object.
#   3. the emitter's manifest method lookup -- DOES NOT. This is the one that
#      refuses bit_length and __round__ before lowering is reached.
#
# So a method reaches lowering when table 1 answers and dies there when table 2
# cannot, and never leaves the emitter when table 3 cannot. The three failures
# above are the same missing fact reported from three depths.
#
# ⛔ Why NOT declare int's methods on bool in builtins.mlir: it is the
# variant-adding shape this project rejects, it would have to repeat every
# method int gains, and `base_names` already states the fact -- nothing reads it.
#
# ⛔ Why NOT reuse the source-class base walk (`classMethodSymbol` in
# ABI/CallableABI.cpp:137, which does exactly this loop over `base_names`):
# it resolves through `classForContract`, which does not find manifest classes.
# The diagnostic at Manifest/Calls.cpp:69 relies on that -- `!classForContract`
# is how it decides a receiver is a stdlib contract rather than compiled code.
# The walk is right; the class it walks is not reachable from there.
#
# ⭐ THE RECEIVER SIDE IS ALREADY DONE. Once resolution reaches int's method,
# a bool receiver has to become an int, and `appendRuntimeSource` widens the
# truth bit as of 2026-08-15 (tests/golden/cases/divmod_of_bool.py pins it).
# That is why divmod(True, 2) works and abs(True) does not: divmod resolves,
# abs does not.
#
# differential: skip refused; the point is the refusal

flag: bool = True
print(flag.bit_length())
