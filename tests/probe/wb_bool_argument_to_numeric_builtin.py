# OPEN. bool reaches int's OPERATORS since 2026-08-14 (tests/golden/cases/
# bool_is_an_int.py) but not int's parameter positions, so the numeric builtins
# still refuse it:
#
#   abs(True) ......... builtins.bool.__abs__ ... has no implementation
#   round(True) ....... the same, __round__
#   float(True) ....... refused
#   max(True, 2) ...... refused
#   divmod(True, 2) ... refused
#
# while `str(True)` is fine (str takes anything) and every operator spelling of
# the same conversion now works: `True + 0`, `-True`, `True < 2`.
#
# ⭐ IT IS THE SAME DEFECT AS `def f(x: float)` REFUSING `f(3)`, which the note
# at TypeSystem.cpp records: a rung of the numeric tower is spelled at the
# operators and not at the argument boundary. That note names what is needed --
# "this check has to ACCEPT the argument and the call site has to CONVERT it,
# and accepting without converting hands the callee an int where its ABI
# expects a float's lanes" -- and the conversion for both rungs now exists as
# `emitIntFromBool` / `emitFloatFromInt`. What is missing is the pairing of the
# acceptance with the conversion at the call site, once, for both rungs.
#
# ⛔ Why NOT widen the promotion into the builtin table instead
# (`kDunderBuiltins` in EmitterCalls.cpp, which is where abs/round/float
# resolve): it repairs three names and leaves `max`, `divmod`, `sum` with a
# start value, and every user function that annotates `int` or `float`. The
# table is a list of builtins, not the boundary the rule belongs at.
#
# differential: skip refused; the point is the refusal
print(abs(True), round(True), float(True))
