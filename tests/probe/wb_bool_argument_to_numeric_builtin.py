# FIXED 2026-08-15, all five. Kept for the readings, because this file was
# wrong three times in a row and each wrong reading predicted a repair that
# measurement rejected.
#
#   float(True) ....... FIXED  tests/golden/cases/float_of_bool.py
#   divmod(True, 2) ... FIXED  tests/golden/cases/divmod_of_bool.py
#   abs(True) ......... FIXED  tests/golden/cases/bool_inherits_int_methods.py
#   round(True) ....... FIXED  tests/golden/cases/round_without_ndigits.py
#   max(True, 2) ...... never broken by the time it was written; the operator
#                       repair of 2026-08-14 had carried it, because max compares
#
# ⛔ READING 1, "the numeric tower is missing at the argument boundary, and the
# repair is to accept the argument and convert it at the call site". Wrong about
# the repair, and the note it inherited that from was wrong for two sessions.
# Converting at a PARAMETER boundary is a wrong answer: CPython prints True for
# `def q(n: int): print(n); q(True)` and a converted argument prints 1. See
# tests/probe/wb_argument_boundary_numeric_tower.py.
#
# ⛔ READING 2, "so the boundary is the manifest ABI adapter instead". Half
# right, and confidently wrong about scope. The adapter really was missing an
# arm -- it could unbox an object into i64, i1 and f64 and not widen a truth bit
# into an int's lanes -- and adding it fixed divmod and NOTHING else, because
# abs never got that far.
#
# ⛔ READING 3, "what is left is that bool inherits no method of int's,
# bit_length included". Right about the inheritance (the manifest index carried
# no base information; the protocol table did) and wrong about the evidence.
# `bit_length` is not declared on int at all, so a plain int is refused too, and
# round(True) was a third cause again: int's __round__ contract made ndigits
# required, and `n: int = 5; n.__round__()` was refused as well.
#
# ⭐ WHAT ACTUALLY HELD: five refusals, four causes, at four different depths --
# a missing emitter arm, a missing ABI arm, a missing base walk, and a missing
# contract arity. Every reading that compressed them into one cause produced a
# repair that measured wrong. The shared symptom was "bool reaches a numeric
# builtin", which is a description of the SYMPTOM and was never a cause.
#
# differential: run all five agree with CPython now
print(float(True), divmod(True, 2), abs(True), round(True), max(True, 2))
