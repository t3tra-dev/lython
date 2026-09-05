# FIXED 2026-09-05 for the SINGLE-TOWER shapes. `half(7)` is 3 and
# `positive(3)` is True, and the golden is
# tests/golden/cases/a_return_a_rung_below_its_annotation.py.
#
# What it was: a RETURN whose value stands a rung below the return annotation.
#   def half(n: int) -> float: return n // 2   ; half(7)
#     "type of return operand 0 ('!py.contract<"builtins.int">') doesn't match
#      function result type ('!py.contract<"builtins.float">')" -- the MLIR
#      verifier, over ordinary Python.
#
# ⭐ WHAT LOCATED IT: every OTHER boundary already answered by keeping the
# value's own type -- a local (`x: float = 3` prints 3), a parameter, and a
# parameter default. The return was the last one reading a numeric annotation
# as the answer rather than as a constraint. `functionSignature` now re-reads
# the body when the annotation is a numeric contract and takes the walked
# result if it is a LOWER rung, guarded by an in-flight set so a recursive or
# mutually recursive function takes the annotation and stops.
#
# ⛔ Converting stays rejected: `print(half(7))` would answer 3.0.
#
# ⛔ AND NOT FOR A `complex` ANNOTATION. `inferExpr` answers `builtins.float`
# for `1.0 + 0.0j`, so `def rotate(z: complex, n: int) -> complex` re-read at
# float emitted a body returning a complex against a float ABI -- "cannot adapt
# builtins.complex return value to callable return ABI 0 of rotate", caught by
# golden scalar_loop_carried_mutate, which is the only program in the corpus
# that passes a complex literal EXPRESSION to a complex parameter. The argument
# specializer refuses to trust the same inference for the same reason.
#
# ⛔ STILL OPEN, and deliberately: a body whose ARMS return different rungs.
#
#     def ratio(a: int, b: int) -> float:
#         if b == 0:
#             return 0
#         return a / b
#
# The honest result is `int | float`, whose rung is -1, so the walk keeps the
# annotation and the program keeps its refusal -- with the MLIR verifier's
# message, which is still a defect of LAYER. The py ABI cannot return a union
# and collapsing it along the tower would print 0 for False. Same wall as
# `def pick(n: int) -> int` reached by `pick(True)`.


def half(n: int) -> float:
    return n // 2


def positive(n: int) -> int:
    return n > 0


print(half(7), positive(3))
