# The worklist loop -- iterate a carried list, build the next one, rebind -- is
# refused:
#
#   owned resource from @LyList_FromLength result 0 reaches function exit
#   without release, transfer, or owned return
#
# ⛔ THE REDUCTION IS THREE LINES AND EVERY NEIGHBOUR COMPILES:
#
#     pending = ["a"]
#     for n in range(2):
#         for task in pending:   # <- iterating the carried list
#             pass
#         pending = []
#
#   without the inner loop ..................... compiles
#   reading it another way (`len(pending)`) .... compiles
#   RETURNING the list at the end .............. compiles (the return
#                                                transfers the token out)
#   the same shape at module scope ............. same refusal
#
# ⭐ WHAT THE IR SAYS. The back edge releases the previous list when the rebind
# replaces it; the loop's EXIT edge releases nothing, so the last one reaches
# the function exit owned. One release is missing, on one edge.
#
# ⛔ NOT the view-forwarding drop, which is the first thing the code points at.
# `insertOwnedBlockArgumentReleases` refuses a candidate whose interior VIEWS
# do not forward across an edge ("callers must drop it (leak-safe)"), and
# iterating the list makes exactly such a view. Teaching `forwardedViews` to
# ignore a view that nothing can name after the edge -- computed by walking the
# successors -- changes nothing here, so the candidate is being dropped
# somewhere else or never formed. That is where the next attempt starts: print
# the candidate set for this function before theorising about which guard
# removed it.
#
# ⛔ AND "leak-safe" IS NOT SAFE HERE. The comment that justifies dropping a
# candidate reasons that a missing release only leaks -- but the affine
# verifier refuses a leak, so the program does not compile at all. A guard
# whose failure mode is a rejected program is not a conservative one.
def run() -> None:
    pending = ["a"]
    for n in range(2):
        for task in pending:
            pass
        pending = []


run()
