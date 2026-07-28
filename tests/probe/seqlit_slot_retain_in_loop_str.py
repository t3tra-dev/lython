# SHIPPED DEFECT (2026-07-28), independent of the sequence-literal source-move
# defect and of the release-placement defect. Valid Python, REFUSED by main
# 4699488 with:
#
#   error: ownership CFG exploration exceeded 20000 states
#          (last: retained=1999 borrowed=0 prev=0 stale=0 group=1 token=1)
#
# Mechanism: `s` is a non-temporary source (read again after the literal), so the
# literal correctly retains it into the slot and does NOT move its token. That
# slot retain sits inside a loop. `verifyResourceOnCFGPaths` counts it into
# `state.retained`, and `retained` is part of the visited-state key, so every
# iteration mints a fresh state and the fixpoint never closes. The borrowed-entry
# walk already skips these retains for exactly this reason; the owned walk does
# not. One side of a symmetric pair was never fixed.
#
# ⚠️ AND THE REFUSAL IS NOT MERELY A BAD MESSAGE. Making the walk converge reveals
# that it had been abandoning the exploration BEFORE reaching a real use-after-
# release finding in neighbouring shapes. So a verifier that exits on resource
# exhaustion has said nothing about the judgements downstream of where it stopped:
# "the verifier passed" and "the verifier REACHED that check" are different
# claims, and only the second licenses any conclusion.
#
# ⛔ The fix is NOT to skip the retain -- see seqlit_single_loop_read_back.py and
# seqlit_literal_elem_read_back.py, which that skip regresses from clean to
# refused. The container's release must DISCHARGE the slot retains it absorbed.
s = "abc"
tlen = 0
for k in range(3):
    t = (s,)
    tlen = len(t)
print(len(s), tlen)  # CPython 3.14: 3 1
