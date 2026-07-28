# ✅ REPAIRED 2026-07-28 (dict side). Kept for the same reason the sequence twin
# is: the pair is what separates "trip count" from "immortal small-int cache" as
# the axis, and any future regression here will look like a threshold again.
#
# This is tests/probe/seqlit_cache_boundary_pair.py transposed onto the dict
# literal, and it reproduced. The dict literal shipped on bcfbbf9 deciding the
# element source-move on `valueIsConsumedOnlyBy` ALONE -- a use-SET fact -- after
# the sequence literal had already been repaired to conjoin an execution-
# FREQUENCY fact. The in-tree comment on that predicate labelled the dict side a
# KNOWN GAP rather than a judgement, because it had never been measured. It has
# now.
#
# Measured on bcfbbf9, 5 reps each:
#   range(0, 3)   trips 3, values 0..2   -> ..... clean
#   range(3, 6)   trips 3, values 3..5   -> XXXXX abort
#   range(-3,-1)  trips 2, small |v|     -> XXXXX abort
# Same trip count, opposite outcome, so the trip count is NOT the axis. Negatives
# fail despite tiny magnitude (they take the heap path), so magnitude is not it
# either -- membership of the immortal cache {0, 1, 2} is. `LyLong_FromI64`
# returns a shared immortal global for exactly those three, which ABSORBS the
# over-release and is the reason this took an extra repair cycle to surface after
# the sequence side closed.
#
# ⛔ Do not touch the cache in builtins.mlir to "fix" this; the cache is doing
# nothing wrong.
cached = 0
for i in range(0, 3):
    for j in range(2):
        a = {"k": i}
        cached += a["k"]
print(cached)  # CPython 3.14: 6

heap = 0
for i in range(3, 6):
    for j in range(2):
        b = {"k": i}
        heap += b["k"]
print(heap)  # CPython 3.14: 24

negative = 0
for i in range(-3, -1):
    for j in range(2):
        c = {"k": i}
        negative += c["k"]
print(negative)  # CPython 3.14: -10
