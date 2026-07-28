# ✅ REPAIRED 2026-07-28. All three blocks now print CPython's values. Kept
# because the pair is what separates "trip count" from "immortal small-int cache"
# as the axis, and any future regression here will look like a threshold again.
#
# The axis of the nested-loop over-release, as a pair that differs only in the
# VALUE the outer loop variable takes -- not in trip count, not in structure.
#
# `LyLong_FromI64` returns an IMMORTAL global for exactly {0, 1, 2}
# (`__ly_long_zero/one/two_*`) and a heap allocation for everything else. An
# over-release of the loop variable is therefore absorbed while it stays inside
# that set, which is why the defect looked like a threshold at n=4: `range(n)`
# first reaches 3 there.
#
# Measured on main 4699488:
#   range(0, 3)   trips 3, values 0..2   -> clean
#   range(3, 6)   trips 3, values 3..5   -> fails
# Same trip count, opposite outcome, so the trip count is NOT the axis. Negative
# values fail too despite tiny magnitude (they take the heap path), so magnitude
# is not the axis either -- membership of the cache is.
#
# A literal element of the same value is clean (`ys = [3]`), so it is the
# PROVENANCE of the value (produced by the outer loop's iterator) combined with
# the frequency mismatch, not the value itself.
#
# ⛔ Do not touch the cache in builtins.mlir to "fix" this; the cache is doing
# nothing wrong. It is the reason the defect took this long to surface.
cached = 0
for i in range(0, 3):
    for j in range(2):
        a = [i]
        cached += a[0]
print(cached)  # CPython 3.14: 6

heap = 0
for i in range(3, 6):
    for j in range(2):
        b = [i]
        heap += b[0]
print(heap)  # CPython 3.14: 24

negative = 0
for i in range(-3, -1):
    for j in range(2):
        c = [i]
        negative += c[0]
print(negative)  # CPython 3.14: -10
