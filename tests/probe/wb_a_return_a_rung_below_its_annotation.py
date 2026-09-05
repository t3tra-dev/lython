# probe: a RETURN whose value stands a rung below the return annotation
# CLASSIFICATION @ 0398ac59: 3 loud, all from the MLIR verifier (defect of LAYER)
#   def half(n: int) -> float: return n // 2   ; half(7)
#     "type of return operand 0 ('!py.contract<"builtins.int">') doesn't match
#      function result type ('!py.contract<"builtins.float">')"
#   def positive(n: int) -> int: return n > 0  ; positive(3)   -- bool vs int
#   def ratio(a, b) -> float: `return 0` on one arm, `a / b` on the other
# CPython 3.14 expects: 3 / True / 0.5 and 0
#
# ⭐ THE SAME QUESTION AT EVERY OTHER BOUNDARY IS ANSWERED, and answered by
# KEEPING the value's own type: a local (`x: float = 3` prints 3), a parameter
# (`f(3)` on `def f(x: float)` -- specialization, wb_argument_boundary_numeric_
# tower.py), and now a parameter DEFAULT (`def go(v: float = 0)` --
# wb_a_default_a_rung_below_its_annotation.py). The RETURN is the boundary that
# still takes the annotation literally, because `functionSignature` sets
# `sig.resultType = annotationType(returns)` without consulting the body.
#
# ⛔ Converting is rejected here for the reason it is rejected everywhere else:
# `print(half(7))` would answer 3.0 where CPython answers 3.
#
# ⛔ And the third shape must stay refused whatever is done to the first two:
# its two arms are different rungs, the honest result is `int | float`, and the
# py ABI cannot return a union. That is the same wall `def pick(n: int) -> int`
# reached by `pick(True)` hits.
#
# The repair is to walk the body when the annotation is a NUMERIC contract and
# take the walked result if it is a lower rung -- with a recursion guard, since
# the annotation is also what breaks the cycle for a recursive function today.


def half(n: int) -> float:
    return n // 2


def positive(n: int) -> int:
    return n > 0


print(half(7), positive(3))
