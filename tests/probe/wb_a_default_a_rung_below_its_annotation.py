# FIXED 2026-09-05. The three shapes below now answer CPython's 1 / 1.5 / True,
# and the golden is tests/golden/cases/a_default_a_rung_below_its_annotation.py.
#
# What it was: a parameter DEFAULT that stands a rung below its annotation.
#   def go(v: float = 0) -> float: return v + 1.5 ; go()
#     "runtime bundle value 0 for '!py.contract<"builtins.float">' has type
#      'i64', but ABI expects 'memref<3xi64>'" -- from the LOWERING.
#
# ⭐ WHAT LOCATED IT: the ARGUMENT boundary of the same question was already
# closed by specialization (wb_argument_boundary_numeric_tower.py), so the two
# spellings of one question disagreed. `recordMonomorphicFunction` listed a
# positional default beside *args/**kwargs/keyword-only as a reason not to
# register the function at all, and the mapping a default actually breaks is
# only the one from OPERANDS: filling the omitted tail from the declared
# parameters instead is exact for a call with no keywords and no `*`.
#
# ⛔ Converting the default to the declared rung stays rejected: `go()` would
# answer 0.0 where CPython answers 0, the measurement that rejected converting
# at the argument boundary too.
#
# ⛔ Literal defaults only. The rung has to be known without emitting the
# expression, and a literal is the only default whose inferred type cannot
# disagree with what emission would produce.


def go(v: float = 1) -> float:
    return v


def rung(v: float = 0) -> float:
    return v + 1.5


def flag(n: int = True) -> int:
    return n


print(go(), rung(), flag())
print(go(2.5), rung(2.5), flag(7))
