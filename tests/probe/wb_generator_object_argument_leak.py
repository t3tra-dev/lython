# UNBOUNDED LEAK on 4699488, fixed in the same change that added this file.
#
# `iter(x)` over a builtin container lowers to a state-machine generator, and
# the creation site persists the iterable into the generator's frame words so
# the drop finalizer can resume the body for its close semantics. It did that
# with BOTH effects at once:
#
#   Ly_IncRef(range)                             ly.ownership.aggregate_retain
#   __ly_generator_frame_store_builtins_range    ly.ownership.transfer_args=[2]
#
# The helper is shared with the frame-lane suspend store, where transfer is
# correct (the value is an owned result of the resume clone and dies at the
# store). At the argument site the value does NOT die -- the Python local stays
# readable -- so the site retains as well. Two references, and the affine
# invariant is then 2 = 1: `produce + retain` on the left, and on the right only
# the drop finalizer's aggregate release, because the declared transfer told
# release placement the creator's own token was already gone.
#
# MEASURED with `leaks --atExit`, baseline (`print(0)`) = 1 root / 540672 B from
# LyRt_InstallStackGuard, subtracted:
#
#   iterations      100      1000     40000
#   excess roots    100      1000     40000      64 B each, no saturation
#
# `tests/probe/tools/leak.py` cannot see this: it has a 500 B/iteration floor
# and this is 64. Golden cannot see it either -- exit 0, stdout correct.
# `releaseaudit.py` on the `refcount-elision` dump of the module-level spelling
# reports `builtins.range(w5)  1A 0R` before and `1A 1R` after.
#
# THREE READINGS THAT THE MEASUREMENTS REFUTED, recorded so they are not
# re-tried:
#
# 1. "LyGenerator_DecRef does not walk the frame slots." It does, since
#    bda1b60: the deallocator's release-to-zero calls
#    `__ly_generator_drop_dispatch`, which routes the storage's target id to a
#    generated per-target finalizer that releases every held frame lane AND
#    every object argument. The finalizer emitted for this program contains
#    `LyRange_DecRef ... "builtins.range:generator drop argument"` and it fires.
#    The defect was never a missing release; it was a second reference.
#
# 2. "Drop the retain and let the transfer stand." Built and measured: the
#    affine-ownership verifier rejects it outright --
#      released owned resource from @LyRange_New is used after release
#      (by call to '...__lyrt_gen_resume__advance')
#    because the resume site borrows the span after the store consumed the
#    token. The retain was load-bearing; only the transfer was wrong. So the
#    leak had been MASKING a use-after-release, and the two are one defect
#    seen from either side.
#
# 3. "Make the frame's hold a borrow so the caller keeps releasing." Not
#    testable here, and NOT shown to be safe: the argument for its unsoundness
#    is that a generator outlives its creating scope, and that shape does not
#    currently compile at all -- `return gen(xs)` is rejected with
#    "runtime manifest has no types.GeneratorType.__next__ method". Recorded as
#    "could not construct the refuting shape", not as "safe".
#
# The fix splits the symbol: `__ly_generator_arg_store_*` carries the retain
# schema (no transfer) and pairs with the borrowing `__ly_generator_arg_load_*`
# the finalizer already used; `__ly_generator_frame_store_*` keeps the transfer
# schema for the lane path.
def f(n: int) -> int:
    total = 0
    i = 0
    while i < n:
        x = range(3)
        it = iter(x)
        total += next(it)
        i += 1
    return total


print(f(4))
