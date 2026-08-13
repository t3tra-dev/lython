# LEAKS the boxed value of a UNION carried local when the loop body never
# runs: 1 allocation / 52 B. This is the EXIT-EDGE half of the union
# loop-carried contract, recorded as still missing when the ENTRY half was
# repaired 2026-08-13
# (tests/probe/wb_union_loop_carried_borrow_overrelease.py).
#
# BISECTED (tests/leak_gate.py against ./build/bin/lyc):
#
#   pick(None, 3) .................. 1 alloc / 52 B   <- this file
#   pick(None, None) ...............  0 (clean -- a None union's release is
#                                     a no-op, so the missing one is invisible)
#   pick(1, 2) .....................  0 (clean -- the body runs and consumes)
#   the same signature with `if` ...  0 (clean -- no loop, no carried local)
#
# THE MECHANISM, already located: `acquireUnionCarriedTokens`
# (EmitterLoops.cpp) retains a union carried local on the loop's ENTRY edge,
# which is the acquisition the ownership pass places for every other type.
# The matching release belongs on the EXIT edge and is placed by
# `insertOwnedBlockArgumentReleases` (Passes/Ownership.cpp), which skips any
# group whose `condition` is set:
#
#     if (!g.deallocator || g.condition)
#       continue;
#
# A union's release is guarded by its tag, so it always has a condition, so a
# union carried local never gets one. The entry retain then stands alone --
# and it stands even when the loop body never executes, which is why the
# ZERO-ITERATION call is the one that leaks.
#
# THE REPAIR is the one that note names: emit a tag-guarded release
# (`cmpi eq(tag, activeTag)`) instead of skipping the group. That pass emits
# only unguarded calls today, which is the whole reason for the skip.
#
# FOUND by tests/probe/tools/leak_sweep.py over tests/golden/cases: 374
# programs, 369 clean, 4 leaking. This shape is two of the four
# (`while_condition_narrowing`, `string_annotation_union`) and is the likely
# attribution for the "bounded, 62 B each, ATTRIBUTION UNKNOWN" family in the
# debt list, which had no reproducer.
#
# differential: skip the leak is invisible to stdout; this records the shape


def pick(v: int | None, other: int | None) -> int:
    seen = 0
    while v is not None:
        seen += v
        v = other
        other = None
    return seen


print(pick(None, 3))
