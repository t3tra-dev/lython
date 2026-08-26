# FIXED 2026-08-15. 1 allocation / 52 B when the loop body never ran: the
# EXIT-EDGE half of the union loop-carried contract, still missing when the
# ENTRY half was repaired 2026-08-13
# (tests/probe/wb_union_loop_carried_borrow_overrelease.py).
#
# BISECTED (tests/leak_gate.py against ./build/bin/lyc):
#
#   pick(None, 3) .................. 1 alloc / 52 B
#   pick(None, None) ...............  0 (clean -- a None union's release is a
#                                     no-op, so the missing one is invisible)
#   pick(1, 2) .....................  0 (clean -- the body runs and consumes)
#   the same signature with `if` ...  0 (clean -- no loop, no carried local)
#
# `acquireUnionCarriedTokens` (EmitterLoops.cpp) retains a union carried local
# on the loop's ENTRY edge, which is the acquisition the ownership pass places
# for every other type. The matching release belongs on the way out, and
# `insertOwnedBlockArgumentReleases` skips any group whose `condition` is set.
#
# ⛔ THE SECOND HALF OF THAT SENTENCE HAS EXPIRED, 2026-08-27: it used to read
# "a union's release is guarded by its tag, so it always has one". It is not
# guarded any more. `forEachActiveUnionMember` retains and releases EVERY
# member rather than the live one, because an inactive member's lanes are the
# immortal dead placeholder and the runtime reads the refcount before anything
# else. The skip is still there, and a conditional group still gets no release
# from that pass -- but a union's refcounting no longer depends on it.
#
# ⭐ THE REPAIR IS THE EMITTER'S, and what settles that is reading
# `carriedLoopEdgeOperands`: it does not only release the lane the body
# replaced, it RE-ACQUIRES the replacement (`acquiringLanes`). The loop's token
# therefore rides the carry, and releasing whatever is carried at the after-block
# balances the entry retain in all three shapes:
#
#   body never runs .... exit carries v0; the release balances the entry retain
#   body keeps v0 ...... no edge release and no reacquire; the same
#   body rebinds to v1 . the back edge released v0 and retained v1, so the exit
#                        release discharges THAT token, while v1's own producer
#                        token stays the pass's to place
#
# ⛔ Why NOT the three placements this file used to reason through, each of
# which was said to break one of the four bisect lines:
#   - retain at the TOP OF THE BODY: still true, a leak per iteration for a lane
#     that keeps its value.
#   - retain immediately before the back edge's release: still true, a no-op
#     pair, which is the over-release the entry half exists to fix.
#   - release on the EXIT EDGE in the emitter: this was the one that was WRONG.
#     It said the release would dangle when the body rebound to a fresh value,
#     because the after-block still names it and the emitter cannot ask whether
#     anything reads it. But the token released there is the loop's, minted by
#     the reacquire, and never the value's only one. The objection was written
#     without reading `acquiringLanes`.
#
# ⛔ Why the after-block START and not the exit edge itself: with a break the
# after-block has several predecessors and each carries the token, so one
# release at the join covers them all where an edge release would need one per
# edge and would still miss the else block's.
#
# ⛔ What was NOT needed after all: seeding a group in the ownership pass and
# teaching release placement to spell a tag guard. Both were scoped
# (`emitterLaneIncrefInBlock` already books the entry edge as a transfer once a
# candidate exists, and `g.condition` is skipped in three separate walks) and
# neither is required, because the guard already exists -- py.decref on a union
# lowers through `forEachActiveUnionMember` exactly as py.incref does.
#
# tests/golden/cases/while_condition_narrowing.py pins it and is now in
# LYTHON_LEAK_GATE_CASES. With this, a sweep of tests/golden/cases is 385
# programs, 384 clean, 0 leaking -- down from 4 leaking on 2026-08-14.
#
# differential: run agrees with CPython now


def pick(v: int | None, other: int | None) -> int:
    seen = 0
    while v is not None:
        seen += v
        v = other
        other = None
    return seen


print(pick(None, 3), pick(None, None), pick(1, 2))
