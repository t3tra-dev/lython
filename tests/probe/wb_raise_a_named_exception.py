# FIXED 2026-08-29 for three of five shapes; the two that remain are below.
#
# WAS: `raise <a name>` failed in every spelling, in three different sentences.
#
#   err = ValueError(m); raise err   (caught) -- "owned resource ... released
#     or transferred more than once on one CFG path". The release sat on the
#     ANCHOR's true edge, which is the unwind path out of the raise itself, so
#     the token had already moved into the callee when that edge materialised.
#     `collectEdgeDeaths` (Passes/Ownership.cpp) now asks that before liveness;
#     the unwind-CLEANUP placement already carried the same exclusion.
#
#   def rethrow(e): raise e -- and -- def fail(msg): raise ValueError(msg) --
#     "borrowed entry argument 0 of @f is released or transferred without a
#     prior retain". The mirror of the retain rule that already existed for a
#     borrowed value RETURNED, now written for a borrowed value CONSUMED
#     (`insertBorrowedConsumeRetains`). Three exclusions were each measured
#     into it by the leak gate: a manifest helper writes its own ownership
#     (`__ly_raise_message_object` is handed a message its caller never
#     releases -- 79 B on io_seek), a generator resume clone is HANDED its
#     exception by throw() (128 B on cross_generator_throw_unwind), and only
#     the emitter's own ABI is what makes a parameter borrowed at all.
#
# STILL OPEN 1 -- reading the name AFTER the handler, which is what this file
# runs. The frame gave its reference away and needs it BACK, so this one wants
# a retain before the raise rather than an exclusion.
#
# ⛔ THREE PLACEMENTS PUT A RELEASE BEHIND THE RAISE, and all three have to be
# stopped before the retain helps. Found by stamping every `emitGroupRelease`
# with `__builtin_LINE()` under an env var and reading the site off the IR --
# which is the tool to reach for here, because guessing cost two rounds:
#
#     afterUseReleases ..... after the last use in the block
#     beforeTermReleases ... before the block's terminator
#     edgeReleases ......... on the terminator's EDGE, and for a `cf.br` that
#                            is written BEFORE the branch, so it reads exactly
#                            like the other two and was the last one found
#
# ⭐ WITH ALL THREE SKIPPED for a block that raises, the release behind the
# raise is gone and the retain is in -- and the affine verifier STILL refuses
# ("released or transferred more than once"). What is left is whether it
# CREDITS that retain: it is an unfold retain with no ownership attribute, on a
# `memref.cast` of a subview of the group's root. That is the next thing to
# read, and it is a question about the VERIFIER rather than the placement.
#
# STILL OPEN 2 -- a union-typed exception:
#
#     for exc in [ValueError("v"), KeyError("k")]:
#         raise exc
#     # runtime manifest has no .raise primitive
#
# Note the EMPTY contract name in that message: the element's bundle carries no
# contract, so neither `manifest.primitive` nor the ancestor fallback can name
# one. A different defect from the two ownership ones.
#
# The three fixed shapes are a golden now
# (tests/golden/cases/an_exception_raised_through_a_name.py), registered in the
# leak gate; this file keeps only what still fails.
held = ValueError("read after the handler")
try:
    raise held
except ValueError as e:
    print("caught held:", e)
print(str(held))

for prepared in [ValueError("v"), KeyError("k")]:
    try:
        raise prepared
    except Exception as e:
        print(type(e).__name__, e)
