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
# So what is left here is not a missing bundle: the ownership placement walk
# does not converge inside the budget on the resume clone's CFG for a nested
# loop, and n=3 is enough. The message names the budget, which is the right
# shape for a refusal, but the walk is the thing to look at -- the same
# nested loop in a NON-generator costs nothing.
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
