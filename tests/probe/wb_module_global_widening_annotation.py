# OPEN, and NEW (found 2026-08-15). A module global annotated with a WIDER
# numeric type than the value assigned to it fails in lowering with a message
# that names nothing the user can act on:
#
#     module global 'x' assignment value group has 3 values, expected 1
#
# BISECTED (./build/bin/lyc, differential against python3.14):
#
#   x: float = 3        at module level ......... internal error   <- this file
#   x: float = y        (y: int), module level .. internal error
#   x: float = 3.0; x = 4 ....................... internal error on the rebind
#   x: float = 3        inside a function ....... 3  (= CPython)
#   x: int = 3          at module level ......... 3  (= CPython)
#   x: float = 3.0      at module level ......... 3.0 (= CPython)
#
# So it is a widening annotation on a MODULE GLOBAL specifically, at any
# assignment, not only the initializer.
#
# ⭐ ROOT CAUSE, and it is the "accept without converting" hazard in the flesh.
# The emitter coerces the value to the declared type, and `coerceValue`
# (EmitterExpressions.cpp) ends in a ClassUpcastOp for any ContractType target:
#
#     %2 = py.int.constant "3" : !py.literal<3>
#     %3 = py.class.upcast %2 : !py.literal<3> -> !py.contract<"builtins.float">
#     py.global.set "x" = %3 {ly.global.boxed} : !py.contract<"builtins.float">
#
# ClassUpcastOp is a RETYPING -- correct for Derived -> Base, where the object
# handle is unchanged, and a lie for int -> float, whose runtime values are
# three lanes and one. `lowerObjectGlobalSet` (GlobalOps.cpp:393) then derives
# the expected lane count from the value's declared type, finds the bundle still
# carrying the int's three, and reports the count. The count is the symptom; the
# upcast is the defect.
#
# The function-local spelling works because a local is typed by its own value
# and never coerced to the annotation -- and prints 3, agreeing with CPython.
#
# ⛔ Why NOT convert here with `emitFloatFromInt`, making the cell a real float:
# it prints 3.0 where CPython prints 3, which the differential buckets WRONG.
# The same measurement that rejects conversion at a parameter boundary rejects
# it here; see tests/probe/wb_argument_boundary_numeric_tower.py.
#
# ⛔ Why NOT refuse in `coerceValue` when the two contracts have different
# representations: it turns an internal error into an honest diagnostic, which
# is worth doing, but it REJECTS `x: float = 3.0; x = 4` -- ordinary Python that
# pyright accepts and CPython runs. Better than an internal error, still a false
# rejection.
#
# ⭐ THE REPAIR IS TO TYPE THE CELL BY WHAT IS STORED, not by the annotation:
# the global's type should be the join of its assigned value types (int | float
# here), which is what the local path gets from SSA for free. Then nothing
# upcasts, both prints agree with CPython, and the annotation goes back to being
# the constraint it is everywhere else.
#
# differential: skip internal error; the point is that it never reaches stdout

x: float = 3
print(x)
