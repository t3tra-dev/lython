# OPEN, and the repair was attempted and REVERTED. A borrowed int parameter
# rebound inside a NESTED loop is refused, and the refusal is provably wrong:
#
#     def f(n: int) -> int:
#         i = 0
#         while i < 2:
#             while n >= 10:
#                 n -= 10
#             i += 1
#         return n
#
#     borrowed entry argument 0 of @f is returned with 2 retained ownership
#     tokens; exactly one may be transferred
#
# The `-> str` spelling of the same shape reaches the other arm with 1
# ("reaches function exit with 1 retained ownership token(s)"). `roman()` --
# the digit-table loop everybody writes -- is this shape, and so is any
# accumulator that consumes its own parameter.
#
# ⭐ THE EMITTED CODE IS CORRECT AND THE COUNT IS WRONG, measured two ways
# rather than argued:
#
#   `lyc jit --release` (verifier off) prints CPython's answer for all four
#   spellings, including roman(1994) == MCMXCIV.
#
#   2000 calls measure at net 0 allocations / 0 B on `tests/leak_gate.py`
#   (LYTHON_LEAK_GATE_LYC_FLAGS=--release), so nothing is leaked or freed
#   twice at run time.
#
# ⭐ AND THE IR SAYS WHY. One `Ly_IncRef ... "block-arg-merge-borrow"` per
# loop entry edge -- ^bb0 for the outer merge, ^bb4 for the inner -- and the
# outer back edge (^bb13) releases the value the inner retain was taken on.
# The pairing is real; what the walk cannot follow is that the release names
# the group's PRE-MERGE name while the state tracks its post-merge one. So the
# balance runs high by one lend per rename, and a nested loop renames twice.
#
# ⛔ THE OBVIOUS REPAIR IS NOT ENOUGH, and this is the useful part. The unwind
# check in the same walk ALREADY exempts this state and says so: "once a merge
# edge renamed the group, the balance includes block-arg-merge lends whose
# paired release targets a pre-merge name -- a state the insertion pass cannot
# discharge, so rejecting it would hard-error plain loop-reassignment code
# (documented residual)". Applying that same exemption at the RETURN sites
# moves the refusal rather than removing it:
#
#     borrowed entry argument 0 of @f retain balance exceeded 64
#
# because the balance keeps climbing around the loop and hits the walk's cap
# before it ever reaches a return. The exemption has to be at the point the
# count stops meaning anything, not at the point it is read.
#
# ⛔ Two mechanisms, and choosing between them is the work:
#
#   END THE PATH at the rename, the way `groupRedefined` already ends one for
#   a back edge that rebinds the merge argument. Cheap, and it stops checking
#   a borrowed parameter after its first rebinding -- which is a real
#   weakening of a memory-safety verifier, not a tidy-up.
#
#   CREDIT A RELEASE UNDER A PRE-MERGE NAME, which is what the code actually
#   does. That means `BorrowedPathState` keeping the group's previous names
#   and deciding which one a release cancels. It is the same forward-only
#   rename family as `wb_generator_resume_raise_unwind.py`, whose four-part
#   repair refused 80 of 490 tests -- but that was `verifyResourceOnCFGPaths`
#   (owned resources) and this is `verifyBorrowedEntryOnCFGPaths`, a smaller
#   and separate walk, so the count there is not evidence about this one.
#
# Reverted rather than shipped because a verifier this one is a root of the
# memory-safety proof, and the first attempt was measured to be wrong.

def f(n: int) -> int:
    i = 0
    while i < 2:
        while n >= 10:
            n -= 10
        i += 1
    return n


print(f(25))
