# ✅ REPAIRED 2026-07-28: the source move is declined when the literal's block can
# reach itself without passing through the source's defining block, and the
# container's compile-time contents evidence is dropped with it (or the read-back
# of an element the container does not own becomes a silent wrong answer). Now
# prints 48 on every rep.
#
# SHIPPED DEFECT (2026-07-28), not yet repaired. Nested loop, container literal.
#
# `initializeSequencePayload` (Passes/Runtime/Core/CollectionPayload.cpp) hands the
# element SOURCE's token to the container when `valueIsConsumedOnlyBy` says this
# literal is the source's only user. That is a use-SET fact standing in for an
# execution-FREQUENCY one: a literal nested in a loop the source is defined
# OUTSIDE of has ONE use that runs many times, so the one token is handed over
# many times.
#
# Emitted at runtime-lowering (BEFORE refcount insertion) as
# `aggregate_release = "builtins.int:sequence.literal.source"` on the OUTER loop's
# value, inside the INNER body, immediately before the inner backedge.
#
# Observed on main 4699488: alternates between `Ly_DecRef observed non-positive
# refcount` (exit 134 / 133) and exit 0 printing `0`. BOTH faces occur for the
# same binary, so an exit-code-only check misses half of it. Survives --release.
#
# The immortal small-int cache is exactly {0, 1, 2} (`__ly_long_zero/one/two_*`,
# builtins.mlir), so the over-release is absorbed while the loop variable stays
# in that set. That -- not the trip count -- is the axis: `range(0,3)` (trips 3)
# is clean and `range(3,6)` (trips 3) is not.
total = 0
for i in range(4):
    for j in range(4):
        ys = [i, j]
        total += ys[0] + ys[1]
print(total)  # CPython 3.14: 48
