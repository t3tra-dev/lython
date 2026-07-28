# ✅ NOW CORRECT (prints 3) after the 2026-07-28 dict-side repair.
#
# ⛔ THE MOST IMPORTANT GUARD IN THE DICT GROUP, and the direct twin of
# tests/probe/seqlit_nested_read_only_silent.py. Read before touching either
# CollectionPayload.cpp's source-move predicate or its evidence demotion.
#
# On bcfbbf9 this program is REFUSED (rc=1), with `ownership CFG exploration
# exceeded 20000 states (last: retained=1110 parked=1 ...)`. A refusal is a safe
# failure -- nothing runs, nothing is mis-executed -- but per rfc/stdlib-semantics
# 13j-3 it is NOT evidence that the checks below it would have passed: the walk
# never reached them.
#
# THE MEASUREMENT THIS FILE EXISTS FOR. Landing the frequency query WITHOUT the
# contents-evidence demotion turns that refusal into `WWXWX` -- exit 0 printing
# 0 instead of 3, on 3 of 5 reps of ONE binary. Reproduce it from the shipping
# binary with:
#
#     LYTHON_ABLATE_DICT_EVIDENCE_DEMOTION=1 lyc jit <this file>
#
# That is the one direction this family may never move in, and it is strictly
# worse than the abort the repair was aimed at, because an abort announces itself
# whereas this is silent and survives --release.
#
# Why the pair does it: declining the move stops the dict from taking the
# source's token, while the literal's compile-time contents evidence still
# resolves `r["a"]` to the already-stored SSA value -- so the reader gets an
# element the container does not own, and `v` outlives the dict.
#
# Corollary for anyone measuring this family: fold rc != 0 AND value-mismatch
# into FAIL when DETECTING, but never when measuring which DIRECTION a change
# moved -- `R`, `X` and `W` are three different outcomes and collapsing them
# hides exactly this transition. Use reps >= 5; single-rep grids miss the
# intermittent faces entirely (measured here: `WWXWX`).
v = 0
for i in range(3, 4):
    for j in range(2):
        r = {"a": i}
        v = r["a"]
print(v)  # CPython 3.14: 3
