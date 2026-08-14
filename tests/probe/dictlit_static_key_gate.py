# FIXED 2026-08-15. It was the LAST program in either corpus that hit the
# affine state cap, and the scan is now zero:
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
# ⭐ AND THE HELPER IS NEVER REACHED, which the code already says out loud. A
# comment in `initializeDictPayload` (Core/CollectionPayload.cpp, beside the
# source-move rule) records it: "A dict literal reaches this function only when
# every key is a `py.str_constant` (PackAndBindingOps.cpp), so `{i: v}` never
# gets here -- one non-static key sends the whole literal down the `setitem_box`
# probe path." That path emits its own `dict.literal.key` / `dict.literal`
# retains and does not call `chargeSlotRetainsToParent`, which lives as a
# file-local in the other file. Hence `aggregate_id=0`: nothing on this path ever
# asks for the container's identity.
#
# That also explains the file's NAME, which predates the diagnosis: the static
# key gate is what routes `{s: 1}` and `{i: 1}` to different lowerings.
#
# ⛔ Not the insertion block, which was the second guess and was measured: the
# early return for a moved insertion block was replaced with a program-order
# walk from the anchor, rebuilt, and `{i: 1}` still stamped nothing -- because
# the helper is not on that path at all. Reverted rather than kept: a codegen
# change that does not do what it was written for is not worth its risk.
#
# ⭐ THE REPAIR was to give `chargeSlotRetainsToParent` a home both lowerings can
# reach and call it from the probe path too. Three localizations, two of them
# measured wrong first, and the third was already written in a code comment.
#
# ⚠️ AND THE WARNING DID NOT COME DUE, which is worth recording as precisely as
# the warning was. The July twin's repair exposed a real use-after-release once
# the walk could reach the checks past the cap. This one exposed nothing: 700
# tests green in both builds, the golden sweep 386 programs with 385 clean and 0
# leaking, and this program itself net zero. The cap had been hiding a correct
# program, not a finding.
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
# differential: run agrees with CPython now

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
