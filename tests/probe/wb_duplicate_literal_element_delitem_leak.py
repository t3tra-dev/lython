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
# WHERE TO LOOK. The literal takes an `aggregate_retain` per slot -- two here
# -- but `movedSources` (Core/CollectionPayload.cpp) dedupes the `.source`
# release to ONE, because the source holds one token to hand over and may only
# be released once.
#
# LOCATED, by diffing the emitted `@f` against the same program with the `del`
# removed (LYTHON_IR_DUMP=refcount-elision). The retains are IDENTICAL in both
# -- four on `%1#0`, the same four ops. What the `del` changes is the exit:
#
#   without del: ... {aggregate_release = "...sequence.literal.source"}
#                LyList_DecRef(%4)                     ; teardown, length 2
#                LyUnicode_DecRef(%1#0) {reference_release}   <- the FRAME's
#
#   with del:    ... {aggregate_release = "...sequence.literal.source"}
#                LyUnicode_DecRef(%1#0) {aggregate_release = "...list.delitem"}
#                LyList_DecRef(%4)                     ; teardown, length 1
#                                                      ; and NO frame release
#
# So the `del` did not add a discharge, it REPLACED one: the walk read the
# delitem release as this group's death (`consumeSites`) and placed no release
# at the function exit. The delitem release discharges the SLOT, so the
# frame's own reference is the one left over. The single-element spelling is
# clean because there the frame has nothing left after the literal takes its
# token; with the source filling two slots it does.
#
# ⛔ Not the unfold-retain count repaired 2026-08-14: with that in place this
# group asks for zero unfold retains (2 consumes, 1 slot release, 2 retains
# held, credit 1), and the leak is 48 B with it and without it.
#
# ⛔ And NOT `consumeSites`, which the exit diff points straight at. Making a
# slot release ineligible to be a group's death -- take the last consume that
# is not one, and record no death when every consume is -- was implemented and
# measured: 48 B, unchanged, and every other case in this family stayed clean.
# So the frame's exit release does not come from that decision, and the thing
# the `del` displaces is placed somewhere else. Reverted rather than left in:
# it changes codegen with no test able to see it.
#
# ⛔ And it is not a different STRATEGY being chosen.
# LYTHON_OWNERSHIP_TRACE_PLACEMENT is 331 lines, byte-identical in order
# between the two programs, and LYTHON_TRACE_OWNED_LOCAL is identical too. So
# every group takes the same arm in both; what differs is where that arm PUTS
# the release, inside `releaseOwnedGroupByLiveness`.
#
# Narrowed to that function, with the surrounding decisions measured equal:
# `readFollowsConsumeInBlock` is true in both (the `xs[0]` read follows), so
# `consumeSites` is empty and `groupHasConsumingCall` false in BOTH, and the
# unfold-retain count is 1 in both. The remaining difference is the liveness
# itself -- `lastUse` / `consumedBlocks` / the edge releases -- over a block
# that now holds two releases of the same name instead of one.
#
# `msg` is TWO groups over one object: the `LyUnicode_FromBytes` call result
# and the `owned_local_object` marker cast of it. Without the del both are
# released at the exit, one after the other; with the del only the marker's
# is. So the arm to instrument next is the call-result group's liveness, not
# the marker's.
#
# differential: skip the leak is invisible to stdout; this records the shape


def f() -> None:
    msg = "hi there"
    xs = [msg, msg]
    del xs[0]
    print(len(xs), xs[0])


f()
