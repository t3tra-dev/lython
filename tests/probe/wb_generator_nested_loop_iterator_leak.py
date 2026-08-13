# FIXED 2026-08-14. Kept as the reproducer, and as the record of what the
# affine walk's non-convergence was hiding.
#
# WAS: 1 allocation / 56 B above the AOT baseline, BOUNDED -- the identical
# figure at n=3, n=10 and n=40, so one range iterator whose refcount never
# reached zero. Invisible until 2026-08-14, because the ownership walk did
# not converge on this CFG and refused with "ownership CFG exploration
# exceeded 20000 states". The ⚠️ note at that cap says a refusal there is not
# a safe-side failure; this is the instance that proves it.
#
# MEASURED, at the time:
#
#   this file, nested loops in a generator ... 1 alloc / 56 B, any n
#   one loop before a yield ..................  0 (clean)
#   the yield inside the loop ................  0 (clean)
#
# THE CAUSE: a token stopped being OWNED when it passed through a forwarding
# block. `isOwnedIncoming` (Passes/Ownership.cpp) recognised a block argument
# only when the argument was itself a merge candidate, so the inner loop's
# back edge saw its own argument -- the outer iterator, forwarded from the
# outer loop's merge through a single-predecessor block -- as unowned and
# LENT it a token every trip. Nothing gave it back. A single loop was clean
# because its back edge carries the merge argument itself, with no forwarding
# block in between.
#
# The repair follows the forwarding edge back to where the token is owned,
# which turns the lend into the transfer it always was. It deliberately stops
# at a block with no predecessor: an entry argument is genuinely borrowed
# (70 of the 299 borrow-edge retains in the 2026-07-30 census are entry
# arguments and not one carries a transfer/release/retain contract).
#
# tests/golden/cases/generator_nested_loops.py pins the value; this file and
# tests/leak_gate.py pin the absence of the leak (net 0 at n=3, 10 and 40).
from typing import Iterator


def f(n: int) -> Iterator[int]:
    total = 0
    for i in range(n):
        for j in range(2):
            total = total + i * j
    yield total


for v in f(3):
    print(v)
