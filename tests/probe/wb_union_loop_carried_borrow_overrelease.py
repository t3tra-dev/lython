# FIXED 2026-08-13 -- kept as the reproducer for the half that remains.
#
# WAS: a silent wrong answer, and sometimes an abort. CPython prints "start";
# this printed an empty line with rc=0 and no diagnostic -- 12 of 12
# sequential runs -- and aborted instead when the differential runner ran it
# under load.
#
# A loop-carried local of UNION type is released on the loop's back edge --
# `py.decref` of the old incarnation, which lowers to a release guarded by the
# union tag -- and on the first iteration the old incarnation is the caller's
# borrowed argument. The release is never compensated, so the caller's string
# is freed while it is still owned there.
#
# MEASURED (RelWithDebInfo + Debug, both, on the pre-audit binary and today):
#
#   this file, `return "start"` ................. "" every run (12/12)
#   `out = "start"` before the loop, return out . aborts 8 of 12 runs with
#                                                 "Ly_DecRef observed
#                                                 non-positive refcount"
#   the same with `return len(out)` ............. aborts 6 of 12 runs
#   `int | None` instead of `str | None` ........ correct (a small int's
#                                                 release is a no-op, so the
#                                                 over-release is invisible)
#   `if s is not None:` instead of `while` ...... correct (no back edge)
#   `while s is not None: break` ................ correct (no back edge taken)
#   a non-union carried local (`while flag:`) ... correct
#
# The nondeterminism is allocator reuse, not a race: the over-release frees an
# object the caller still reads, and whether that shows up as a wrong value or
# as a refcount abort depends on what is allocated into the hole. The wrong
# value below is the stable half, which is why THIS spelling is the one in the
# corpus -- the aborting spellings would make the differential runner flaky.
#
# THE REPAIR: the loop acquires its own token for a union carried local on
# the entry edge (acquireUnionCarriedTokens, EmitterLoops.cpp), which is the
# acquisition the pass places for every other type. The exit-edge RELEASE is
# still missing, so the second reproducer below still refuses.
#
# MECHANISM, located: insertOwnedBlockArgumentReleases
# (src/lython/lowering/Passes/Runtime/Passes/Ownership.cpp) places both halves
# of the loop-carried contract -- the borrow-edge retain that compensates a
# back-edge release, and the release on the exit edge -- and it skips any
# group whose `condition` is set. A union's release is conditional on its tag,
# so a union-carried local gets NEITHER half. The missing retain is this file.
# The missing exit release is the other symptom of the same hole:
#
#   def g(n: int | None) -> int:      # refused, does not reach execution:
#       total = 0                     # "owned resource from @LyLong_Add
#       while total < 3:              #  result 0 reaches function exit
#           total += 1                #  without release, transfer, or owned
#           n = total + 100           #  return"
#       return total
#
# WHAT IT UNBLOCKED: `while x is not None:` now narrows x for its own body.
# That narrowing is one line in emitWhile, and before the repair it made this
# worse rather than better -- the unwrap moved which lanes the pass sees and
# turned "aborts 6 of 12" into "aborts every run". With the token acquired,
# all five aborting spellings are 0 of 12 and the narrowing is in.
#
# tests/golden/cases/while_condition_narrowing.py is the golden for both.
def f(s: str | None) -> str:
    while s is not None:
        s = None
    return "start"


print(f("ab"))
