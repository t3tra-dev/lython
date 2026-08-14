# OPEN, narrowed three times on 2026-08-15. This file started as "five
# refusals, one cause"; both halves were wrong, and one refusal is left.
#
#   abs(True) ......... FIXED (tests/golden/cases/bool_inherits_int_methods.py)
#   divmod(True, 2) ... FIXED (tests/golden/cases/divmod_of_bool.py)
#   float(True) ....... FIXED (tests/golden/cases/float_of_bool.py)
#   max(True, 2) ...... already worked; the operator repair of 2026-08-14
#                       carried it, because max compares
#   round(True) ....... refused, and NOT for a bool reason: int's __round__
#                       contract makes ndigits required, so round(True, 0) works
#                       and `n: int = 5; n.__round__()` is refused as well. See
#                       tests/probe/wb_manifest_class_inherits_nothing.py.
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
# ⭐ AND SO IS THE RESOLUTION. abs(True) needed a second half the adapter could
# not supply: the manifest index had to find int's `__abs__` under a bool
# receiver. `selectManifestMethod` now walks the base chain. divmod landed first
# only because divmod resolved and abs did not -- the two halves were
# independent all along, which is why the earlier "one cause" reading kept
# producing predictions that measured wrong.
#
# ⛔ Why NOT widen the promotion into the builtin table (`kDunderBuiltins` in
# EmitterCalls.cpp), which is where this file first pointed: it would have
# papered over two names at the one boundary that turned out to hold neither
# half of the cause. The table is a list of builtins, not a boundary.
#
# differential: skip refused; the point is the refusal
print(round(True))
