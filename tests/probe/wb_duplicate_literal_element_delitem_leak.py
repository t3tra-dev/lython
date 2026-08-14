# FIXED 2026-08-15. 1 allocation / 48 B, and it PREDATED the aggregate-retain
# counting repair of 2026-08-14 -- measured identically on a binary built before
# it. The shape is one source filling TWO slots of a literal, then a `del`:
#
#   xs = [msg, msg]; del xs[0] ....... 1 alloc / 48 B
#   xs = [msg];      del xs[0] .......  0 (clean)
#   xs = [msg, msg]; no del ..........  0 (clean)
#   xs = ["p", "q"]; del xs[0] .......  0 (clean -- two distinct sources)
#
# ⭐ THE LEDGER, counted in @f at refcount-elision, is what finally named it.
# `msg` is TWO frame groups over one object: the `LyUnicode_FromBytes` call
# result and the `owned_local_object` marker cast of it.
#
#   without del:  DecRef(%1#0) {aggregate_release = "...sequence.literal.source"}
#                 DecRef(%1#0) {reference_release}          <- the second group
#                 LyList_DecRef(%4)                         ; teardown, 2 slots
#                 = 4 discharges
#
#   with del:     DecRef(%1#0) {aggregate_release = "...sequence.literal.source"}
#                 DecRef(%1#0) {aggregate_release = "...list.delitem"}
#                 LyList_DecRef(%4)                         ; teardown, 1 slot
#                 = 3 discharges
#
# The delitem does not ADD one: it removes a slot, so the teardown shortens by
# exactly the amount the delitem adds. What went missing is the second frame
# group's exit release, because the walk read the delitem as its death.
#
# ⭐ THE RULE: of the `aggregate_release` labels, `.source` is the one that
# discharges the moved value's OWN token; every other spelling discharges a
# CONTAINER's reference, and a slot's discharge is never a frame token's death.
# It is not caught by `consumeIsAnotherReferencesDischarge` because that
# predicate's third leg asks whether the name is ours and here it IS -- the
# delitem releases `%1#0`, a group value, since a slot and the frame reference
# one object. Only the label separates them.
#
# ⛔ Why the rule is guarded by "the `.source` release is PRESENT" rather than
# the label alone: "any non-`.source` aggregate release is never a death" was
# implemented and measured, and 185 of 384 goldens stopped compiling with
# "owned resource from @LyLong_FromI64 result 0 is released or transferred more
# than once on one CFG path". A slot release is the only discharge a group has
# in plenty of shapes. What makes this one different is that the value's own
# token has already LEFT: after `.source` moves it into the container, the
# frame's remaining reference belongs to the other group.
#
# ⛔ Not the unfold-retain count repaired 2026-08-14: with that in place this
# group asks for zero unfold retains, and the leak was 48 B with it and without.
#
# ⛔ Not `consumeSites`, which the exit diff points at. Making a slot release
# ineligible to be a group's death -- take the last consume that is not one --
# was implemented and measured at 48 B unchanged. Nor is it leg 1 of the
# discharge predicate: dropping `isMinted` entirely, measured 2026-08-15, is
# also 48 B, because leg 3 was the one answering.
#
# ⛔ And not a different STRATEGY. LYTHON_OWNERSHIP_TRACE_PLACEMENT is 331
# lines, byte-identical in order between the two programs, and
# LYTHON_TRACE_OWNED_LOCAL is identical too.
#
# tests/golden/cases/duplicate_element_delitem.py pins it, red-checked at 48 B
# on the pre-fix binary, and is in LYTHON_LEAK_GATE_CASES.
#
# differential: run agrees with CPython now


def f() -> None:
    msg = "hi there"
    xs = [msg, msg]
    del xs[0]
    print(len(xs), xs[0])


f()
