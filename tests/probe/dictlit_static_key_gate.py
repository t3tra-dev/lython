# OPEN, and the LAST program in either corpus that hits the affine state cap:
#
#     ownership CFG exploration exceeded 20000 states
#     (last: retained=952 parked=0 borrowed=0 prev=0 stale=0 group=1 token=0)
#
# MEASURED 2026-08-15 over tests/golden/cases + tests/probe, 717 programs: this
# one and no other. That figure matters because the cap is not a diagnostic --
# the note at AffineOwnership.cpp says so outright, and says any claim that the
# verifier is green on a program requires that the program did not hit it. The
# exposure is now a number instead of an unknown.
#
# ⭐ ROOT CAUSE, and it is ONE MISSING STAMP. The chain, measured end to end:
#
#   1. `LyDict_FromLength` carries no `ly.ownership.aggregate_id`. Counted in
#      the emitted `__main__`: the dict program has 0 of them, the equivalent
#      list program has 2.
#   2. So `aggregateIdentityOf(container lane)` answers nothing, and
#      `chargeSlotRetainsToParent` (Core/CollectionPayload.cpp) returns without
#      stamping -- it IS called on the dict path, right after the
#      `dict.literal.key` and `dict.literal.value` retains.
#   3. So those retains carry `aggregate_retain` and no `aggregate_parent`,
#      and `slotRetainParent` in the verifier answers nothing.
#   4. So the walk counts them in `state.retained` instead of parking them in
#      `state.slotParents`. `retained` is part of the visited-state key and
#      `slotParents` is bounded by the number of containers, so inside nested
#      loops the fixpoint never closes. `parked=0` in the message above is that
#      fact printed.
#
# This is the same shape as tests/probe/seqlit_slot_retain_in_loop_str.py,
# repaired 2026-07-28 by charging the retain to the container's identity: one
# side of a symmetric pair, again, and this time the side that was never
# stamped rather than the side that never read the stamp.
#
# ⚠️ MAKING IT CONVERGE MAY EXPOSE A REAL FINDING, and that is not a reason to
# leave it. The 2026-07-28 repair of the sequence-literal twin revealed a
# genuine `used after release` in neighbouring shapes that had been invisible
# the whole time the cap was firing. Budget the repair with room to chase what
# it uncovers.
#
# differential: skip refused; the point is the refusal

probe = 0
for i in range(3, 6):
    for j in range(2):
        d = {i: 1}
        for k in d:
            probe += k
print(probe)  # CPython 3.14: 24

payload = 0
for i in range(3, 6):
    for j in range(2):
        e = {"k": i}
        payload += e["k"]
print(payload)  # CPython 3.14: 24
