# OPEN, and NEW with the union renderer of 2026-08-14 (the values are right;
# this is the accounting half). 1 allocation / 42 B when the member that
# renders is a STR. The int member is clean, and so is the same union never
# rendered.
#
# BISECTED (tests/leak_gate.py against ./build/bin/lyc):
#
#   print(pick_str(True))  -> "hi" ......... 1 alloc / 42 B   <- this file
#   print(pick_int(True))  -> 7 ............  0 (clean)
#   v = pick_str(True); print(v is None) ...  0 (clean -- not rendered)
#   print(pick_str(True), pick_str(False)) .  0 (clean -- the two-argument
#                                            path joins through __add__, so
#                                            nothing hands the payload out)
#
# THE ASYMMETRY, and it is the same one the renderer is built on: every member
# but str builds a NEW string, so the union's payload and the rendered value
# are different objects. A str member renders to ITSELF -- the first arm of
# `emitStringifyValue` is the identity -- so the rendered value IS the union's
# payload, reached through `py.union.unwrap`.
#
# THE LEDGER, from the emitted `__main__` (LYTHON_IR_DUMP=refcount-elision):
#
#   %4:5 = call @pick(%true)          ; tag, member lanes, owned-result lanes
#   ^bb1:                             ; the str member is live
#     cf.br ^bb4(%4#1, %4#2)          ; forwarded, NOT released
#   ^bb2:                             ; the None member is live
#     LyUnicode_DecRef(%4#3)          ; the union's obligation, discharged
#     ...renders the "None" literal
#   ^bb4(%8, %9):
#   ^bb5:
#     LyUnicode_DecRef(%8)            ; the rendered string
#
# ⭐ ROOT CAUSE, and the ^bb1 line above is missing from that sketch: the merge
# takes a BORROW RETAIN there --
#
#   ^bb1:
#     Ly_IncRef(%cast_2) {aggregate_retain = "block-arg-merge-borrow"}
#     cf.br ^bb4(%4#1, %4#2)
#
# -- so the str path reads: "hi" born with 1, +1 for the merge borrow, -1 at
# ^bb5. It ends at 1. The union's own obligation, the one
# `ly.ownership.owned_results` names, is never discharged on that path.
#
# And the merge is right to retain, GIVEN WHAT IT KNOWS. A union return carries
# the payload TWICE -- once as the member lane (%4#1) and once as the
# owned-result evidence lane (%4#3) -- and the callee returns the same value
# for both (`return %c0_i64, %1#0, %1#1, %1#0, %1#1`). Only %4#3 is in
# `ownedValues`, because only %4#3 is what the ABI declares owned. So
# `isOwnedIncoming(%4#1)` says borrowed, the edge lends, and nothing gives the
# lend back.
#
# ⛔ Retaining the payload in the renderer, on the reasoning that the caller
# releases what it is handed: implemented and measured, 42 B before and 42 B
# after, and the dumps are IDENTICAL apart from the retain's label
# (`builtins.str:py.incref` where the merge would have written
# `block-arg-merge-borrow`). The pass credited the emitter's retain instead of
# adding its own -- which is `emitterLaneIncrefInBlock` doing exactly its job.
# Reverted; the renderer is not the side that is unbalanced.
#
# ⭐ SO THE REPAIR IS TO TELL THE CALLER THE TWO LANES ARE ONE OBJECT. The
# callee knows -- it returns one value into both -- and nothing in the ABI
# says so, so the caller's ownership walk cannot know. An
# `owned_result_aliases`-style note beside `ly.ownership.owned_results`, read
# where the call-result groups are seeded, would put %4#1 in `ownedValues` and
# turn the lend into the transfer it already is.
#
# ⛔ NOT by un-aliasing the two lanes in the callee: for an int member they
# were two objects until 2026-08-14 and that was the union return's double
# materialization, 52 B per call. One object is right; the caller just has to
# be told.
#
# tests/golden/cases/union_renders_by_tag.py pins the VALUES, which are
# correct, and is deliberately NOT in LYTHON_LEAK_GATE_CASES because of this.
#
# differential: skip the leak is invisible to stdout; this records the shape


def pick_str(flag: bool) -> str | None:
    if flag:
        return "hi"
    return None


print(pick_str(True))
