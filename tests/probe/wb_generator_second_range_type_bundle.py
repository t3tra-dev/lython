# PARTLY FIXED 2026-08-13. The type-object half is repaired; what this file
# still reproduces is a BUDGET, reached only by nested loops.
#
# WAS: a generator body that builds a second `range` -- nested loops, or one
# loop after another -- stopped at "new class object has no lowered type
# bundle", which names a lowering invariant and not anything the reader
# wrote. The resume clone's SSA normalization threads every value that
# crosses a block boundary through a new block argument, and a
# `!py.type<...>` was among them; that type has no runtime ABI, so nothing
# could ever give the argument a bundle. Type objects are rematerialized in
# the using block now (GeneratorStateMachine.cpp).
#
# MEASURED after the repair:
#
#   two sequential loops, one yield at the end .... correct
#   two sequential loops, the yield in the second . correct
#   both are in tests/golden/cases/generator_loop_before_yield.py
#   NESTED loops, one yield at the end ............ this file: refused with
#       "ownership CFG exploration exceeded 20000 states", a stated budget
#   two sequential loops with a yield in EACH ..... refused, honestly:
#       "source generator next lowering currently supports only straight-line
#       pure int yield bodies"
#
# So what is left here is not a missing bundle, and it is NOT A BUDGET
# SHORTFALL either. Measured by raising kMaxAffineStates to 4,000,000: the
# walk still does not close, and the counter that grows is `borrowed` --
# 1,428 at the 20,000-state cap, 285,714 at four million. It is unbounded.
#
#   state.borrowed is incremented at a block-argument merge borrow retain
#   (AffineOwnership.cpp, `isBlockArgMergeBorrowRetain`) and decremented by a
#   release through a PRE-RENAME name. It is part of the visited-state key,
#   so a borrow retain on a cyclic path whose cancelling release is not on
#   that path makes the key differ every trip and the fixpoint never closes.
#
# That is the same failure mode the ⚠️ note at the cap already records for
# `retained` (tests/probe/seqlit_slot_retain_in_loop_str.py), and the comment
# at the increment site names both counters as having shown it. Which means
# this refusal MAY BE MASKING A REAL FINDING: the cap is not a safe-side
# failure, and nothing downstream of where the walk stopped has been judged.
#
# The same nested loop in a NON-generator costs nothing -- the resume clone's
# CFG is what puts a borrow retain on a cycle.
#
# differential: skip the refusal is the recorded state, not a wrong answer
from typing import Iterator


def f(n: int) -> Iterator[int]:
    total = 0
    for i in range(n):
        for j in range(2):
            total = total + i * j
    yield total


for v in f(3):
    print(v)
