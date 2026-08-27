# probe: a container built in an inner loop from an outer loop's local
# axes: acquire=retain width=int op=container flow=nested-loop observe=regression
# CLASSIFICATION @ 2026-08-28: 1 正しい
#
# `ys[0]` folds back to the literal's source -- `i`, which is another name for
# the OUTER loop header's argument -- and mints a retain-rooted owned-local
# token for the read-back. The mint is in the INNER loop's body; the source's
# other uses are spread over the outer loop, so the alias-class liveness read
# "live to the end of the outer loop" and wrote the release on the outer loop's
# dead edges, which the mint does not dominate:
#
#   error: operand #0 does not dominate this use
#
# Two things had to change, and the ONE that looked most obviously needed was
# not among them:
#
#   1. the placer must not read a use the mint does not DOMINATE as this token's
#      liveness, and must not write a release where the mint does not reach.
#      With both restricted to the mint's dominated region the release lands
#      right after the `LyLong_Add` that reads the token, which is where it
#      belongs.
#   2. the affine walk's RELEASED arm charged a slot-absorption retain to
#      `retained`, which its live arm hands to the CONTAINER instead. `retained`
#      is part of the visited-state key, so a container built in a loop climbed
#      one per trip and the fixpoint never closed: "ownership CFG exploration
#      exceeded 20000 states (retained=1816)".
#
# ⛔ AND NOT the retain that mints the marker, which looked like the same kind of
# mis-attribution -- the release that pays for it names the MARKER and the walk
# does not credit it to the source, so charging the retain to the source seemed
# to be the imbalance. Excluding it turned the shape into "released or
# transferred more than once on one CFG path": the retain is what the source's
# own later release is spent against. Excluding the release instead, by marker
# shape, broke 9 goldens (`zip_strict`, `unpack_arity`, `user_defined_iterator`,
# ...) with "still owned when ... may unwind" -- a marker minted on a call
# RESULT is a second name for one reference, not a second reference. Both were
# built and measured before being taken back out.
#
# There is no `range` in this program. `for i in range(n)` reached it once the
# loop was written out as a counter, which is how it was found.
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
