# OPEN. 1 allocation / 48 B, and it PREDATES the aggregate-retain counting
# repair of 2026-08-14 -- measured identically on a binary built before it
# (wb_aggregate_slot_unfold_retain_leak.py is that repair; this shape survived
# it, which is why it gets its own file rather than a line in that one).
#
# The shape is one source filling TWO slots of a literal, then a `del`:
#
#   xs = [msg, msg]; del xs[0] ....... 1 alloc / 48 B   <- this file
#   xs = [msg];      del xs[0] .......  0 (clean)
#   xs = [msg, msg]; no del ..........  0 (clean)
#   xs = ["p", "q"]; del xs[0] .......  0 (clean -- two distinct sources)
#
# So it needs the DUPLICATE and it needs the delete: neither alone shows it.
#
# WHERE TO LOOK, from the ledger the two clean lines fix. The literal takes an
# `aggregate_retain` per slot -- two here -- but `movedSources`
# (Core/CollectionPayload.cpp) dedupes the `.source` release to ONE, because
# the source holds one token to hand over and may only be released once. So
# the entity should stand at 1 (its own) + 2 (slots) - 1 (source) = 2 after
# the literal, and the `del` plus the teardown of the surviving slot should
# take it to 0. One of those three discharges is not happening.
#
# ⛔ Not the unfold-retain count: with the repair in place this group asks for
# zero unfold retains (2 consumes, 1 slot release, 2 retains held, credit 1),
# and the leak is unchanged at 48 B either way.
#
# differential: skip the leak is invisible to stdout; this records the shape


def f() -> None:
    msg = "hi there"
    xs = [msg, msg]
    del xs[0]
    print(len(xs), xs[0])


f()
