# FIXED 2026-08-15. 1 allocation / 42 B when the member that rendered was a
# STR; the int member was clean, and so was the same union never rendered.
#
# THE ASYMMETRY, and it is the one the renderer is built on: every member but
# str builds a NEW string, so the union's payload and the rendered value are
# different objects. A str member renders to ITSELF -- the first arm of
# `emitStringifyValue` is the identity -- so the rendered value IS the union's
# payload, reached through `py.union.unwrap`.
#
# ⭐ ROOT CAUSE: a union return hands its payload back TWICE -- once as the
# member lane (%4#1) and once as the owned-result evidence lane (%4#3) -- and
# the callee returns one value into both (`return %c0_i64, %1#0, %1#1, %1#0,
# %1#1`). `staticEvidenceCoveredLogicalOffsets` already knew they were the same
# object: it is what SKIPS the member lane's group, so that only one release is
# placed. But the skip left the member lane in no group at all, so
# `isOwnedIncoming(%4#1)` said borrowed, the merge took a
# `block-arg-merge-borrow` retain, and nothing gave the lend back:
#
#   ^bb1:                              ; the str member is live
#     Ly_IncRef(%cast_2) {aggregate_retain = "block-arg-merge-borrow"}
#     cf.br ^bb4(%4#1, %4#2)
#   ^bb2:                              ; the None member is live
#     LyUnicode_DecRef(%4#3)           ; the union's obligation, discharged
#   ^bb5:
#     LyUnicode_DecRef(%8)             ; the rendered string
#
# "hi" was born with 1, +1 for the merge borrow, -1 at ^bb5, and ended at 1.
#
# ⭐ THE REPAIR WAS THE FACT, NOT A GROUP. `staticEvidenceDuplicateLanes` names
# the covered lanes and the refcount pass records them as owned; no second
# group, so no second release. The corpus went 382/383 clean to 383/383 clean
# apart from while_condition_narrowing, and
# tests/golden/cases/union_renders_by_tag.py joined the leak gate.
#
# ⛔ Why NOT retain the payload in the renderer, on the reasoning that the
# caller releases what it is handed: implemented and measured, 42 B before and
# 42 B after, with dumps IDENTICAL apart from the retain's label
# (`builtins.str:py.incref` where the merge would have written
# `block-arg-merge-borrow`). The pass credited the emitter's retain instead of
# adding its own -- `emitterLaneIncrefInBlock` doing exactly its job. The
# renderer was never the unbalanced side.
#
# ⛔ Why NOT un-alias the two lanes in the callee: for an int member they WERE
# two objects until 2026-08-14, and that was the union return's double
# materialization, 52 B per call. One object is right; the caller just had to
# be told.
#
# differential: run agrees with CPython now


def pick_str(flag: bool) -> str | None:
    if flag:
        return "hi"
    return None


print(pick_str(True))
print(pick_str(False))
