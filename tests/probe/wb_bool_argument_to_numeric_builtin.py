# OPEN, narrowed twice on 2026-08-15. This file started as "five refusals, one
# cause"; both halves were wrong, and what is left of it is two refusals whose
# cause is not numeric.
#
#   abs(True) ......... refused: builtins.bool.__abs__ has no implementation
#   round(True) ....... refused: bool does not provide manifest '__round__'
#   divmod(True, 2) ... FIXED (tests/golden/cases/divmod_of_bool.py)
#   float(True) ....... FIXED (tests/golden/cases/float_of_bool.py)
#   max(True, 2) ...... already worked; the operator repair of 2026-08-14
#                       carried it, because max compares
#
# ⭐ IT IS NOT THE ARGUMENT BOUNDARY, which is what this file used to say. See
# tests/probe/wb_argument_boundary_numeric_tower.py: at a PARAMETER boundary
# CPython keeps the argument's own type and converting is a wrong answer. At the
# MANIFEST ABI it is the opposite -- CPython's bool inherits int's `__abs__` and
# `__round__` unchanged and they return an int, so `abs(True)` is 1, and the
# native implementation reads the value numerically where nothing can observe
# that it stopped being a bool. Converting there is the operation, not a
# divergence.
#
# ⭐ AND THE CONVERSION IS NOW THERE. `appendRuntimeSource`
# (Runtime/Calls/Operands.cpp) widens a truth bit into an int's lanes by routing
# it through a lazy primitive-i64 bundle, so the box and its ownership come from
# the arms the int case already runs, and `canAppendRuntimeSource` carries the
# matching arm because it runs first during overload selection. That is what
# fixed divmod, and it is why nothing further is needed on the receiver side.
#
# ⭐ WHAT STILL REFUSES abs AND round IS RESOLUTION, NOT ADAPTATION: bool
# inherits no method of int's in either the manifest implementation index or the
# emitter's lookup, `bit_length` included. Recorded with the three disagreeing
# tables in tests/probe/wb_manifest_class_inherits_nothing.py. divmod works and
# abs does not for exactly that reason -- divmod resolves and abs does not.
#
# ⛔ Why NOT widen the promotion into the builtin table (`kDunderBuiltins` in
# EmitterCalls.cpp): it would paper over two names and leave `bit_length` and
# every other inherited method refused. The table is a list of builtins, not the
# boundary the rule belongs at.
#
# differential: skip refused; the point is the refusal
print(abs(True), round(True))
