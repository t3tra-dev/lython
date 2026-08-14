# OPEN, narrowed. Re-measured 2026-08-15 against python3.14; the previous
# version of this file claimed five refusals and one cause, and both were wrong.
#
#   abs(True) ......... refused: builtins.bool.__abs__ has no implementation
#   round(True) ....... refused: bool does not provide manifest '__round__'
#   divmod(True, 2) ... refused: cannot adapt builtins.bool to runtime input 0
#                                of builtins.tuple.builtin_divmod
#                                [values: 'i1', expected 'memref<2xi64>']
#   float(True) ....... FIXED (tests/golden/cases/float_of_bool.py)
#   max(True, 2) ...... already worked; the operator repair of 2026-08-14
#                       carried it, because max compares
#
# ⭐ IT IS NOT THE ARGUMENT BOUNDARY, which is what this file used to say. See
# tests/probe/wb_argument_boundary_numeric_tower.py: at a PARAMETER boundary
# CPython keeps the argument's own type and converting is a wrong answer. Here
# it is the opposite -- CPython's bool inherits int's `__abs__` and `__round__`
# unchanged and they RETURN an int, so `abs(True)` is 1 and `round(True)` is 1.
# Converting the receiver is not a divergence, it is the operation.
#
# ⭐ THE CAUSE IS INHERITED-METHOD DISPATCH ACROSS A REPRESENTATION CHANGE.
# `py.class @bool` declares nine methods and `__abs__` is not among them
# (runtime/modules/builtins.mlir:209); it arrives through `base_names = ["int"]`.
# The declaration is therefore found, and the implementation is then looked for
# under `ly.runtime.contract = "builtins.bool"`, where only `LyLong_Abs` under
# "builtins.int" exists. The two contracts do not share a representation -- a
# bool is one i1, an int is three lanes -- so the usual base upcast, which is a
# retyping, cannot bridge them.
#
# The asymmetry between abs and round is the same hole seen twice: `abs` reaches
# its method through the builtin dispatcher (LyBuiltin_Abs, builtin_lowering =
# "method"), which walks the bases and finds the declaration, and `round`
# resolves as a direct manifest method lookup, which does not. One of the two
# is wrong about inheritance and they disagree in opposite directions.
#
# ⛔ Why NOT declare __abs__/__round__/__divmod__ on bool with bool receivers:
# it duplicates int's implementations at bool's representation, which is the
# variant-adding shape this project rejects, and it would still be wrong for
# every other method bool inherits.
#
# ⛔ Why NOT widen the promotion into the builtin table (`kDunderBuiltins` in
# EmitterCalls.cpp): it repairs abs and round and leaves divmod, which fails one
# layer down in ABI adaptation rather than in method resolution. The table is a
# list of builtins, not the boundary the rule belongs at.
#
# differential: skip refused; the point is the refusal
print(abs(True), round(True))
