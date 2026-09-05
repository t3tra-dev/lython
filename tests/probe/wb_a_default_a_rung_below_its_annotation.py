# probe: a parameter DEFAULT that stands a rung below its annotation
# CLASSIFICATION @ 82993ec1: 3 loud, all from the LOWERING (a defect of layer)
#   def go(v: float = 0) -> float: return v + 1.5 ; go()
#     "runtime bundle value 0 for '!py.contract<"builtins.float">' has type
#      'i64', but ABI expects 'memref<3xi64>'"
#   def go(n: int = True) -> int: return n ; go()
#     the same sentence with 'i1'
#   def go(v: float = 1) -> float: return v ; go()
#     the same sentence
# CPython 3.14 expects: 1.5 / True / 1  (the default keeps its own type)
#
# ⭐ The ARGUMENT boundary of this exact question is closed by specialization
# (wb_argument_boundary_numeric_tower.py): `f(3)` on `def f(x: float)` emits a
# body per ground signature and answers CPython's 6, not 6.0. A call that OMITS
# the parameter never reaches that decision, because the decision is made on the
# emitted OPERAND types and an omitted argument has none -- so the declared
# float ABI is used and the default, carried as `callable_default_values =
# [{kind = "int", value = "1"}]`, is materialised as an i64 against it.
#
# ⛔ Converting the default to the declared rung is the repair this must NOT
# take: `print(go())` would answer 0.0 where CPython answers 0, which is the
# measurement that rejected converting at the argument boundary too.
#
# The fix is the same mechanism one step over: an omitted parameter whose
# default is a LITERAL of a lower rung should key the specialization at that
# literal's type, exactly as passing it would.


def go(v: float = 1) -> float:
    return v


def rung(v: float = 0) -> float:
    return v + 1.5


def flag(n: int = True) -> int:
    return n


print(go(), rung(), flag())
print(go(2.5), rung(2.5), flag(7))
