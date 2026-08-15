# FIXED 2026-08-15, and by NONE of the four parts costed below.
#
# ⭐ THE REPAIR ACCEPTS THE TWO ENTITIES AND PAYS FOR THE SECOND. Everything
# under "ATTEMPTED REPAIR" tries to make the block-argument incarnation stop
# existing (B+D delete the pure-renaming group, C teaches the verifier the old
# name, A pins the handler check to every name) -- four parts, 80 of 490 tests
# refused, not shipped. Two entities sharing one allocation is not the defect:
# it is the ordinary shape of a loop-carried merge, and it is CORRECT as soon
# as the entry edge LENDS a reference, which is what
# `insertBlockArgMergeBorrowRetains` was already trying to do. It was refusing
# instead, because `borrowEdgeRetainIsSpellable` asked whether the cell's
# refcount word is written AT THE `memref.alloc` -- and the retain does not go
# there. It goes before the branch in the block the alloc dominates, blocks
# later, with `ly.ownership.owned_local_object` in between. The predicate now
# takes the point the retain will occupy (`prefixIsInitializedBefore`).
#
# ⭐ SO THE THREE OBSERVABLE DEFECTS BELOW WERE ALL ONE MISSING RETAIN. Defect 1
# (the pad releases the cell, the handler reads it freed) and defect 2 (two
# releases on the normal path) are both a refcount of 1 being spent twice; with
# the lend it is 2. Defect 3 (the verifier reports no release for a token
# released two instructions earlier) is the one that says so directly -- it
# honours a release under a pre-rename name exactly when `state.borrowed > 0`,
# which is the retain, so the walk it was written to accept is the one that now
# happens.
#
# ⛔ THE INTERVENING STATE IS WHY THIS TOOK A SECOND PASS. Between bcfbbf9 and
# the fix the crash had already become a REFUSAL -- same site, same cause, and
# no longer a memory-safety event, so nothing here read as urgent. A defect
# that has been de-fanged is still the defect.
#
# golden: tests/golden/cases/loop_in_try_cell_merge.py (red-checked; nine
# spellings, `for`/`while`, `except`/`finally`, function and module scope,
# int/float/str, and it is in the leak gate because the repair ADDS a retain).
#
# ============================================================
# SHIPPED SIGSEGV on bcfbbf9. NO GENERATOR, NO EXCEPTION EVER RAISED.
#
# Run it: `rc=139`, deterministic 5/5. CPython prints 3.
#
# This file exists because the axis recorded in
# `wb_generator_resume_raise_unwind.py` NO LONGER HOLDS. That file's
# minimisation table (taken on a frozen older compiler) says the trigger needs
# "a `for` over a state-machine generator", and lists
#
#     same try/except/accumulator, NO generator ............ OK
#
# On bcfbbf9 that row is CRASH. Re-measured with a 4-bucket classifier
# (`.` matches CPython / `W` silent wrong / `R` refused by the compiler /
# `X` runtime abort), 5 reps each, one binary:
#
#   for over generator, exception raised, handler reads accumulator ... XXXXX
#   for over generator, NO exception, handler reads accumulator ....... XXXXX
#   for over LIST LITERAL, NO exception, handler reads accumulator .... XXXXX  <- this file
#   for over generator, NO exception, handler returns a constant ...... RRRRR
#   for over generator, NO try/except at all .......................... .....
#
# So the generator is IRRELEVANT and so is raising: what is required is
#
#   (a) a `for` loop, over anything, and
#   (b) a `try`/`except` in the same function, and
#   (c) a local written in the loop body and READ in the handler.
#
# (c) is what selects between the two failure directions: a handler that reads
# the local crashes, a handler that does not is REFUSED ("owned resource ... is
# still owned when a call to 'LyLong_FromI64' may unwind out of the function").
# Both come out of the same place.
#
# ============================================================
# ROOT CAUSE (measured, not inferred): one object, two ownership entities
# ============================================================
# `for` puts the loop-carried locals in a cell (`memref<16xi64>`), allocated
# ONCE in the entry block and threaded through the loop header as a block
# argument. In the shipped IR:
#
#     ^bb6:  cf.br ^bb7(%alloc, %alloc_46 : ...)     <- entry edge
#     ^bb7(%54: memref<16xi64>, %55: memref<16xi64>) <- loop header
#     ^bb11: cf.br ^bb7(%54, %55 : ...)              <- back edge, self-forward
#
# One allocation. One refcount. But the ownership machinery tracks TWO groups
# with the same deallocator: the op-rooted one (`%50:2`, the
# `ly.ownership.owned_local_object` cast of `%alloc/%alloc_46`) and the
# block-argument one (`%54/%55`). THREE consumers then handle the rename in the
# forward direction only, and each one is a separate observable defect:
#
# 1. `insertUnwindCleanupReleases` / `groupUsedOnHandlerPath`
#    (`lowering/Passes/Runtime/Passes/Ownership.cpp`). Measured with the
#    `LYTHON_UNWIND_TRACE=1` instrument added in this commit:
#
#      dealloc=__ly_dealloc___ly_cell_1 root=op       def=bb0 handler_owns=yes
#      dealloc=__ly_dealloc___ly_cell_1 root=blockarg def=bb2 handler_owns=no
#
#    The op-rooted group is correctly spared -- its `useSites` include the
#    handler's blocks. The block-argument group's `useSites` are the LOOP BODY
#    ONLY (34 uses, all in bb3/bb4), because the handler reads the cell under
#    the pre-loop name. So the pad releases it, then branches to the `except`
#    block, which loads the freed cell and retains the loaded header:
#    `Ly_IncRef(null)`, `EXC_BAD_ACCESS address=0x0`.
#
# 2. `insertOwnedBlockArgumentReleases` (same file) gives the block-argument
#    incarnation its OWN normal-path release at loop exit. The shipped IR for
#    `wb_generator_resume_raise_unwind.py` contains THREE
#    `__ly_dealloc___ly_cell_1` calls in `@f`, and the normal path runs two of
#    them: `^bb12` releases `(%54,%55)`, then `^bb13 -> ^bb14 -> ^bb15 -> ^bb24`
#    releases `(%50#0,%50#1)` -- the same cell, twice, on the path where the
#    loop simply finishes. The deallocator is refcounted
#    (`LyObject_ReleaseStorageToZero` then `memref.dealloc`), so the second call
#    READS a freed header. It is benign today only because that read does not
#    come back as 1; `for over generator, NO try/except` above is that path, and
#    it prints the right answer.
#
# 3. `verifyResourceOnCFGPaths` (`verifier/runtime/AffineOwnership.cpp`) renames
#    the tracked group forward across the loop edge and keeps the old names in
#    `state.previous` -- but it only honours a release written under a
#    `previous` name when that release cancels a borrow-edge retain
#    (`state.borrowed > 0`). A pure renaming has no borrow retain, so once
#    defects 1 and 2 are repaired this walk reports
#    "owned resource ... reaches function exit without release" for a token that
#    is demonstrably released two instructions earlier.
#
#    Note the direction: the verifier stayed quiet on the shipped compiler
#    ONLY BECAUSE the pad freed the cell under the name it was tracking. The
#    double free was what satisfied it. That is 13j-3's shape again -- the cover
#    came off and the body was underneath.
#
# ============================================================
# ATTEMPTED REPAIR: NOT SHIPPED. Four parts, all default OFF.
# ============================================================
# Each part is an env toggle on ONE binary (`lyc` does not rebuild
# byte-for-byte, so differing hashes cannot establish that two arms differ --
# 13j-7), and each is named for what it ENABLES (13j-4):
#
#   A  LYTHON_EXP_BLOCKARG_HANDLER_PINS=1  pin the handler-path liveness check
#                                          to EVERY name of the object
#   B  LYTHON_EXP_RENAMED_ARG_GROUPS=1     do not create an ownership group for
#      + D                                 a pure-renaming block argument, and
#                                          do not count a forward into one as a
#                                          transfer
#   C  LYTHON_EXP_PREV_NAME_RELEASE=1      honour a release written under a
#                                          pre-rename name of the tracked group
#
# Ablation, one binary, probe program + a 5-test discriminator set:
#
#   | A | B+D | C | this file | full ctest (490)      |
#   |---|-----|---|-----------|-----------------------|
#   | 0 |  0  | 0 |  X  crash | 490/490  (shipped)    |
#   | 1 |  0  | 0 |  R  refused | 490/490             |
#   | 1 |  1  | 0 |  R  refused | 4/5 discriminators refused |
#   | 1 |  0  | 1 |  R  refused | 1/5 discriminators refused |
#   | 1 |  1  | 1 |  .  prints 3 | 410/490 -- 80 REFUSED |
#   | 0 |  1  | 1 |  .  prints 3 | (same 80)           |
#
# All five minimisations above become `.` with A+B+C+D on. And the cost is 80 of
# 490 tests REFUSED -- every one in the refusal direction, none silently wrong.
# The refusals are the next step, not a mystery: with the pure-renaming group
# gone, `releaseOwnedGroupByLiveness` has not been taught where the SOURCE
# group now dies, so the source's normal-path release is never placed
# ("owned resource from @LySet_FromLength result 0 reaches function exit
# without release"). Part A alone is 490/490 green and turns the crash into a
# refusal; it is still off, because trading a crash for a refusal of the same
# program is not a fix and the corpus is a numerator, not a denominator (13j).
#
# ⚠️ Do NOT measure this family with `leaks --atExit` before checking the
# process exit status: a crashed run produces no report at all and a naive
# parser reads that as "0 leaks" (13j-9). Do NOT use symbol breakpoints to argue
# a release "never ran": inlining removes them (13j-10).
def f() -> int:
    total = 0
    try:
        for v in [1, 2]:
            total += v
    except ZeroDivisionError:
        total += 100
    return total


print(f())
