# OPEN, the RESIDUE of two repairs. A guard over a FIELD now narrows it: the
# read is CHECKED against the proof -- a test, a branch to a raise, then the
# unwrap or the refine -- which is what makes it sound where a plain view would
# not be. `is None` (cases/a_guard_over_a_field), a base-class field narrowed
# by `isinstance` to a subclass, and an OPTIONAL field narrowed by `isinstance`
# all work.
#
# WHAT IS LEFT is a union field of two REAL members -- `int | str`, `A | B` --
# narrowed by isinstance:
#
#     class Node:
#         def __init__(self) -> None:
#             self.payload: int | str = 0
#         def render(self) -> str:
#             if isinstance(self.payload, str):
#                 return self.payload.upper()
#             return "num"
#     # union<int, str> does not provide manifest method 'upper'
#
# ⛔ MEASURED AND GATED OFF, not merely untried. With the arm accepting it, the
# program compiles and then fails the ownership verifier:
# "ly.ownership.owned_local_object marks a value this frame never acquired: it
# is not a fresh allocation". The unwrap hands back a value the walk books as
# owned and the frame never acquired -- the fragile invariant
# [[lython-fragile-invariants]] names as ownership across narrowing. A verifier
# failure is a WORSE answer than the emit refusal it replaces, so the arm takes
# a union only where it is `T | None`, the shape measured clean.
#
# ⛔ The FALSE side of the same guard is gated off with it. `int | str` minus
# `str` is `int`, and recording it made the same verifier failure appear on
# programs whose true branch alone had been fine -- which is why the else arm of
# a union-field isinstance still reads the whole union.
#
# ⛔ AND THE LOCAL SPELLING IS THE WORKAROUND, as it was for the whole family:
# `v = self.payload; if isinstance(v, str):` compiles and prints CPython's
# answer. The refusal does not say so, deliberately: the message is shared by
# every union read and only this one has that workaround.
#
# Measured 2026-09-04.
class Node:
    def __init__(self) -> None:
        self.payload: int | str = 0

    def render(self) -> str:
        if isinstance(self.payload, str):
            return self.payload.upper()
        return "num"


n = Node()
print(n.render())
