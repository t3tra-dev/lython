# FIXED the silent half (2026-09-02); this is the refusal that remains.
#
# WAS: a wrong answer with no diagnostic. `classify(-1) is False` printed
# False where CPython prints True, and `is not False` printed True where
# CPython prints False. The disjointness fold in the `is` emission asked
# whether the WHOLE type on each side was assignable to the other, and
# `union<int, literal<False>>` is assignable to `bool` in neither direction
# even though one of its members IS a bool -- so a fold whose whole premise is
# "these can never be the same object" fired on two types that share a member.
#
# The fold now expands each side into its widened members and folds only when
# every cross pair is disjoint, so the program below reaches the
# reference-identity path and its refusal:
#
#   `is` requires reference-typed operands that resolve statically; this
#   operand combination has no stable identity
#
# which is the same answer the same program already got when the union was
# spelled in an annotation instead of inferred.
#
# ⛔ WHAT REMAINS: `is` against a union member is a tag test and a lane
# comparison -- `UnionTestOp(bool)` and then the member's value -- which is
# exactly what `emitNoneIdentityTest` already does for the None member, minus
# the value half (None has one inhabitant, so the tag IS the answer). Only a
# member with no lanes gets that treatment today.
#
# ⛔ AND AN ERASED object STILL LIES HERE, separately: when the join reaches
# `builtins.object` instead of a union (`list[int] | dict[str, int]` does),
# the operands are two contracts and a real address comparison is emitted --
# `f(-1) is xs` for `xs` the very list that was returned printed False.
def classify(n: int):
    if n < 0:
        return False
    return n * 2


print(classify(-1) is False, classify(3) is False)
