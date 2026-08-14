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
# ⭐ IT IS THE INT PAYLOAD, NOT THE DICT. Bisected by payload type, counting the
# attributes in the emitted `@f`:
#
#   s = "k"; d = {s: 1} .... aggregate_id=2  aggregate_parent=2  retains=2
#   i = 5;   d = {i: 1} .... aggregate_id=0  aggregate_parent=0  retains=2
#
# So `chargeSlotRetainsToParent` (Core/CollectionPayload.cpp) stamps the dict
# correctly whenever the payload is an ordinary object, and stamps NOTHING when
# it is an int. `aggregate_id=0` says the identity was never even minted, so the
# helper took one of its two early returns before reaching `aggregateIdentityOf`
# -- the container lanes being empty, or the builder's insertion block no longer
# being the block captured before the retains. An int arrives as a LAZY box and
# materialising one emits the fast/slow `scf.if`, which is the one thing in this
# path that can move the insertion block, so that is where to look first.
#
# ⛔ An earlier version of this note said `LyDict_FromLength` carries no
# `ly.ownership.aggregate_id` and that the list twin does. Both halves were
# wrong: the count was taken from the int program only, and the dict gets its id
# on the str spelling. Recorded because the wrong reading is the plausible one
# -- the failing program is a dict, and the working comparison was a list, so
# the container looked like the variable when the payload was.
#
# The consequence downstream is unchanged and still measured: unparented retains
# land in `state.retained` instead of `state.slotParents`, `retained` is part of
# the visited-state key while `slotParents` is bounded by the container count,
# and inside nested loops the fixpoint never closes. `parked=0` in the message
# above is that fact printed. Same shape as
# tests/probe/seqlit_slot_retain_in_loop_str.py, repaired 2026-07-28 by charging
# the retain to the container's identity.
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
