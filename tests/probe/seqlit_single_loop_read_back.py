# ✅ STILL CLEAN after the 2026-07-28 repair, and the guard below held: the fix
# charges the slot retain to its holder rather than skipping it, so the read-back
# stays legal.
#
# ⚠️ AND IT IS STILL A LATENT USE-AFTER-FREE. `LYTHON_EXP_HOLDER_DISCHARGE=1`
# turns on the holder-release discharge and this program is then refused with
# `used after release (by call to 'LyLong_Add')` -- correctly: the emitted order
# is `Ly_IncRef(i)` +1, `LyList_DecRef(ys)` -1 (the deallocator drops the slot),
# `LyLong_DecRef(i)` -1 (the source move), and only then the read. The refcount
# reaches zero one call before the use. It prints 12 because the block has not
# been reused. The discharge is off by default because 23 golden cases have this
# shape; enabling it needs the container's release moved past the reads its slots
# handed out.
#
# ⛔ GUARD PROBE. This program is CLEAN on main 4699488 and must stay clean.
#
# It exists to stop a specific repair that looks right and is not: making the
# affine walk SKIP slot-absorption retains (`aggregate_retain`) the way the
# borrowed-entry walk does (AffineOwnership.cpp, `kAggregateRetainAttr`).
#
# That skip is tempting because `retained` is part of the visited-state key, so
# counting a slot retain inside a loop makes the key increase every iteration and
# the fixpoint never closes -- a real shipped defect (see
# seqlit_slot_retain_in_loop_str.py). But the two walks track different resource
# kinds, and the asymmetry is not a licence to copy the exemption.
#
# THIS is the counterexample: reading the element BACK out of the container hands
# the reader a token derived from the slot, and the walk needs that retain to
# justify the reader's later release. With the retain skipped, this ordinary
# SINGLE loop is refused with
#   `released owned resource ... is used after release (by call to 'LyLong_Add')`
# Measured: 9 of the first 40 golden cases were newly refused by that skip.
#
# Note there is no nesting and no cross-loop borrow here, which is why it was
# absent from the first matrix taken of this defect -- and why that matrix read as
# "improvement on every row". The rule learned: when a matrix backs a shipping
# decision, the first question is which rows are MISSING, not whether the present
# rows agree.
#
# The actual requirement is a modelling change, not a predicate flip: the
# container's release must DISCHARGE the slot retains it absorbed
# (`aggregate(parent, path)` answered by `parent`).
total = 0
for i in range(3, 6):
    ys = [i]
    total += ys[0]
print(total)  # CPython 3.14: 12
