# FIXED 2026-08-14. Kept as the reproducer and the bisect.
#
# WAS: an element removed from or replaced in a container was never released.
# One defect, three figures that had been recorded as three:
#
#   del a[i] ................ 1 alloc, 41 B (str) or 52 B (int)
#   holder[0] = a ........... 3 allocs / 8316 B  (`container_shared_with_a_holder`)
#   grid[1][0] = 9 .......... 1 alloc / 52 B     (recorded in CollectionPayload.cpp)
#
# BISECTED (tests/leak_gate.py against ./build/bin/lyc), which is what showed
# they were one:
#
#   a = ["p", "q"]; del a[0] ......... 1 alloc / 41 B   <- this file
#   the same with no del ............. 0 (clean)
#   a = [1, 2, 3, 4]; del a[3] ....... 1 alloc / 52 B
#   a = [1, 2, 3, 4]; del a[1] ....... 0 (clean -- the shift leaves the last
#                                     slot holding a duplicate, and the two
#                                     halves happened to cancel)
#   d = {"x": 1, "y": 2}; del d["x"] . 1 alloc / 41 B   (the key, same shape)
#
# THE CAUSE, counted off the emitted IR for `["p", "q"]` with and without the
# `del`. The element "p" has TWO references -- its own creation token and the
# `aggregate_retain` the list literal minted for the slot -- and the `del`
# gives it two releases: `sequence.literal.source` (the token the literal took
# over) and `list.delitem` (the slot's). Two for two, so nothing is needed.
#
# `releaseOwnedGroupByLiveness` (Passes/Ownership.cpp) counted "one reference
# in hand, one taken away per consume", saw two consumes against one reference,
# and inserted an UNFOLD RETAIN before the first -- which nothing discharges:
#
#     %cast_11 = memref.cast %5#0 ...
#     call @Ly_IncRef(%cast_11) {aggregate_retain = "builtins.str:..."}  ; +1 slot
#     ...
#     call @Ly_IncRef(%cast_24)                                          ; +1 UNFOLD
#     call @LyUnicode_DecRef(%5#0) {aggregate_release = "...source"}     ; -1
#     call @LyUnicode_DecRef(%5#0) {aggregate_release = "...delitem"}    ; -1
#
# leaving refcount 1 at exit. WITHOUT the `del` there is one consume, the rule
# inserts nothing, and the arithmetic came out right by accident -- which is
# why the whole leak gate stayed green over it.
#
# The repair counts the aggregate retains that are actually in hand instead of
# assuming one. It is byte-identical wherever the count IS one, which is every
# program that was already correct.
#
# ⛔ Why NOT read the second release as "not a read" in
# `readFollowsConsumeInBlock`, which is the other place an unfold retain is
# decided: it is not that path. A deallocator carries `release_args`, so both
# releases are CONSUMES and the retain comes from the all-but-the-last rule.
# Skipping releases there would leave this leak and break the group that has
# one consume and a genuine later read.
#
# Pinned by tests/golden/cases/delete_item.py and
# tests/golden/cases/container_shared_with_a_holder.py in
# LYTHON_LEAK_GATE_CASES.
#
# differential: skip the leak is invisible to stdout; this records the shape

a = ["p", "q"]
del a[0]
print(len(a))
