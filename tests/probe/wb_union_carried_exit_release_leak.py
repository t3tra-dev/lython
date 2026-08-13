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
# THE REPAIR IS NOT THE ONE THAT NOTE NAMES, and the emitted IR is what says
# so. Read at refcount-elision, `@pick` is:
#
#     scf.if %0 { Ly_IncRef(%arg5) {aggregate_retain = "builtins.int:py.incref"} }
#     scf.if %1 { Ly_IncRef(%arg1) {aggregate_retain = "builtins.int:py.incref"} }
#     cf.br ^bb1(%5, %arg5, %arg6, %arg7, ..., %9, %arg1, %arg2, %arg3)
#   ^bb1(%11, %12,%13,%14, %15,%16,%17, %18, %19,%20,%21):
#     ...
#   ^bb4:                       ; the loop exit
#     return %15, %16, %17, ...      ; `seen` only -- nothing released
#
# Two facts fall out. First, the guard is ALREADY THERE: `py.incref` on a
# union lowers through `forEachActiveUnionMember`, so the acquisition is a
# tag-guarded call and the release would lower the same way. Teaching the pass
# to spell a guard is not what is missing.
#
# Second, and this is the actual obstacle: this group is never SEEDED.
# `insertOwnedBlockArgumentReleases` collects candidates from owned CALL
# RESULTS and from `owned_local_object` markers. The reference here is minted
# by an `Ly_IncRef` the EMITTER emitted, which is neither -- it has no
# results, so `collectOwnedCallResultGroups` cannot name it, and it carries an
# `aggregate_retain` label, which books it against a slot rather than the
# frame. So the `g.condition` skip is not what keeps the release away; the
# walk never has a group to skip.
#
# ⭐ Which makes the repair "give the emitter's acquisition a producer the
# pass can see", and only then the guarded release. The tag is available where
# the release has to go -- it rides the same edge as the member lanes and is
# block argument %11 (for `other`) and %18 (for `v`) at ^bb4 -- so the guard's
# operand is not the hard part either.
#
# ⛔ Why NOT move the acquisition instead, which looks like it would avoid all
# of this. Three placements were reasoned through against the four bisect
# lines above and each breaks one of them:
#   - retain at the TOP OF THE BODY rather than the entry edge: correct for a
#     lane the back edge releases, and a leak per iteration for a lane that
#     KEEPS its value (`carriedLoopEdgeOperands` releases only the lanes that
#     changed), so `pick(1, 2)` leaks where it is clean today.
#   - retain immediately before the back edge's release: a no-op pair, which
#     is the same as not releasing at all -- and that is the over-release
#     `acquireUnionCarriedTokens` exists to fix
#     (wb_union_loop_carried_borrow_overrelease.py).
#   - release on the EXIT edge in the emitter: correct when the loop exits
#     with a borrowed value, dangling when the body rebound to a fresh one --
#     the after-block still names it and the emitter cannot ask whether
#     anything reads it. That question is the liveness the pass computes,
#     which is why the release belongs there and not in the emitter.
#
# FOUND by tests/probe/tools/leak_sweep.py over tests/golden/cases: 374
# programs, 369 clean, 4 leaking. This shape is ONE of the four,
# `while_condition_narrowing`.
#
# ⛔ `string_annotation_union` leaks the same 52 B off a union and is NOT this
# defect -- it needs no loop at all, and it stayed red after this shape's
# reproducer was written. It was the union RETURN's double materialization
# (wb_union_return_double_box_leak.py), fixed 2026-08-14. Equal figures off
# the same type are not one attribution; two of them here were three
# defects.
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
