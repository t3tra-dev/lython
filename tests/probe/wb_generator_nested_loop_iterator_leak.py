# LEAKS one range iterator: 1 allocation / 56 B above the AOT baseline,
# BOUNDED -- the identical figure at n=3, n=10 and n=40, so it is one object
# whose refcount never reaches zero, not a per-iteration leak.
#
# This program is the reason the affine ownership walk's non-convergence
# mattered. It used to be refused with "ownership CFG exploration exceeded
# 20000 states", and the ⚠️ note at that cap says a refusal there is NOT a
# safe-side failure -- nothing downstream of where the walk stopped has been
# judged. It was masking exactly this.
#
# MEASURED (tests/leak_gate.py against ./build/bin/lyc):
#
#   this file, nested loops in a generator ... 1 alloc / 56 B, any n
#   one loop before a yield ..................  0 (clean)
#   the yield inside the loop ................  0 (clean)
#
# LOCATED, in the IR the verifier walks (LYTHON_IR_DUMP=refcount-elision):
# the INNER loop's back edge lends the OUTER range iterator a token on every
# trip and nothing gives it back.
#
#     ^bb14(... %43: memref<5xi64>):            ; %43 is the OUTER iterator
#       %subview = memref.subview %43[0][2][1]
#       %cast = memref.cast %subview
#       call @Ly_IncRef(%cast) {block-arg-merge-borrow}
#       cf.br ^bb11(..., %43)                   ; the inner loop header
#
# The outer loop's own back edge (^bb16 -> ^bb4) lends the same way and IS
# balanced, by the `LyRangeIterator_DecRef(%9#4)` the `next` call's result
# gets each trip. The inner loop has the equivalent release for the INNER
# iterator (^bb15) and none for the outer one.
#
# So the lend half of the loop-carried contract was placed and the release
# half was not -- the mirror image of the union-carried-local defect fixed
# 2026-08-13, where a conditional release meant NEITHER half was placed.
# Ownership.cpp's edge-retain loop is where the lend is decided
# (`EdgeRetain`, gated on `insertReleases`), so the two halves already share
# a caller; what is not established is why this candidate group got one.
#
# NOT in the leak gate: a red test is not something to commit
# (tests/CMakeLists.txt). tests/golden/cases/generator_nested_loops.py pins
# the VALUE, which is correct; this file pins the leak.
from typing import Iterator


def f(n: int) -> Iterator[int]:
    total = 0
    for i in range(n):
        for j in range(2):
            total = total + i * j
    yield total


for v in f(3):
    print(v)
