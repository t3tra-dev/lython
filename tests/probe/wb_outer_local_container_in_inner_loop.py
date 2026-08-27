# probe: a container built in an inner loop from an outer loop's local
# axes: acquire=retain width=int op=container flow=nested-loop observe=refusal
# CLASSIFICATION @ 2026-08-28: 3 誤って拒否する
#
# `ys[0]` folds back to the literal's source -- `i`, which is another name for
# the OUTER loop header's argument -- and mints a retain-rooted owned-local
# token for the read-back. The mint is in the INNER loop's body; the source's
# other uses are spread over the outer loop, so the alias-class liveness reads
# "live to the end of the outer loop" and writes the release on the outer loop's
# dead edges, which the mint does not dominate:
#
#   error: operand #0 does not dominate this use
#
# Three separate things have to change for this to compile, and each was built
# and measured before being taken out again:
#
#   1. the placer must not treat a use the mint does not DOMINATE as this
#      token's liveness, and must not write a release where the mint does not
#      reach. Restricting both to the mint's dominated region produces the right
#      IR -- the release lands right after the `LyLong_Add` that reads the
#      token -- and no other golden moves.
#   2. the affine walk's RELEASED arm charges a slot-absorption retain and a
#      marker's root to `retained`, which its live arm does not; `retained` is
#      part of the visited-state key, so a container built in a loop climbs one
#      per trip and the fixpoint never closes ("exploration exceeded 20000
#      states (retained=1816)").
#   3. the release that pays for the marker names the MARKER, and the walk
#      credits it to the source group as well, so the source's own release then
#      reads as "released or transferred more than once on one CFG path". The
#      machinery for this exists -- `callReleasesForeignAggregate` and
#      `own::ReferenceMap` -- and is gated on `hasOwnNamedRelease`, which is
#      false here. Excluding the release by MARKER SHAPE instead broke 9
#      goldens (`zip_strict`, `unpack_arity`, `user_defined_iterator`, ...) with
#      "still owned when ... may unwind": a marker minted on a call RESULT is a
#      second name for one reference, not a second reference, and its release
#      is the producer's.
#
# There is no `range` in this program. `for i in range(n)` reached it only
# because the loop is written out as a counter now, and the counter's shape was
# chosen to bind the target from a value the BODY defines for that reason --
# see `tryEmitLazyIteratorFor`.
#
# CPython 3.14 expects: 24


total = 0
a = 0
while a < 4:
    i = a
    a = a + 1
    b = 0
    while b < 4:
        b = b + 1
        ys = [i]
        total += ys[0]
print(total)
