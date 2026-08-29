# An exception ESCAPES its handler when a `try` body branches and BOTH arms
# call the same raising builtin. CPython prints "caught 1"; this aborts with
# the ValueError uncaught.
#
# ⛔ THE REDUCTION IS SHARP, and every neighbour works:
#   - one arm only                      -> caught
#   - the same two arms with no `try`   -> both raise, as they should
#   - two arms calling DIFFERENT raisers (math.factorial / math.isqrt) -> caught
#   - the try at MODULE scope with two arms                           -> caught
#   - a loop around a try with ONE call                               -> caught
# What is left is: two call sites of the SAME callee inside one try.
#
# ⛔ THE MECHANISM, read off the IR. Each `int()` call is lowered inside its own
# nested traceback-cleanup try, so the raising call is preceded by
# `LyEH_TryCallSiteMarker(<inner id>)` and `LyEH_TryCatchAnchor(<inner id>)`.
# With two identical arms, a block merge inside the `convert-to-llvm` phase
# folds the two blocks and promotes the id to a BLOCK ARGUMENT:
#
#     ^bb5(%43: i64, ...):
#       llvm.call @LyEH_TryCallSiteMarker(%43)
#
# and Cleanup/EH.cpp's `i64ConstantArgument` answers nullopt for it, so the
# call is never registered as a try call site. It keeps only the cleanup's
# unwind edge and the user's handler is never reached. The pipeline already
# records this hazard for HANDLER entries (`createEHSafeCanonicalizerPass`);
# this is the same hazard one level down, and that canonicalizer is Normal --
# it does not merge -- so the merge comes from one of the conversion passes
# beside it.
#
# ⭐ TWO REPAIRS TO WEIGH: give each id its own marker SYMBOL so the blocks are
# not identical and cannot merge, or teach the EH pass to split a block whose
# marker id is a block argument. The first is smaller; it depends on every
# marker call being erased, which the pass already does.
def run(i: int) -> None:
    try:
        if i == 1:
            int("zz")
        else:
            int("yy")
        print("no error")
    except ValueError:
        print("caught", i)


run(1)
