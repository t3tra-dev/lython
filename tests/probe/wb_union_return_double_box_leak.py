# FIXED 2026-08-14. Kept as the reproducer and the bisect.
#
# WAS: 1 allocation / 52 B per call of a function whose declared return type is
# `int | None`. Found by tests/probe/tools/leak_sweep.py as
# `golden.cases.string_annotation_union`, and MIS-ATTRIBUTED at first to the
# union loop-carried contract (wb_union_carried_exit_release_leak.py) because
# both leak 52 B off a union. They are different defects: that one needs a
# loop, this one needs none.
#
# BISECTED (tests/leak_gate.py against ./build/bin/lyc):
#
#   def pick(f: bool) -> int | None: return 7 ..... 1 alloc / 52 B  <- this
#   the same with `-> str | None` and "ab" .......  0 (clean)
#   the same with `-> int` and 7 .................  0 (clean)
#   pick(False) -- the None arm ..................  0 (clean)
#   the quoted annotation `"int | None"` .........  1 alloc / 52 B (same)
#   a union LOCAL, never returned ................  0 (clean)
#
# THE CAUSE, visible in one dump. A union return carries TWO lane groups: the
# union's own member lanes, and the "static returned object evidence" lane that
# `ly.ownership.owned_results` names -- and Returns.cpp requires them to alias
# ("the active member holds the owned token, so the owned evidence lane must
# alias its values"). For `str` they did:
#
#     %1:2 = call @LyUnicode_FromBytes(...)
#     return %c0_i64, %1#0, %1#1, %1#0, %1#1        ; one object, twice
#
# For `int` they did not:
#
#     %0:3 = call @LyLong_FromI64(%c7_i64)          ; the union's member lanes
#     %2:3 = call @LyLong_FromI64(%c7_i64)          ; the evidence lane
#     return %c0_i64, %0#0, %0#1, %0#2, %2#0, %2#1, %2#2, %c7_i64, %true
#
# because an int reaches the wrap as a LAZY box -- raw i64, no object yet.
# `appendUnionRuntimeValues` materializes one to fill the member lanes, but
# `lowerUnionWrap` recorded `*input`, the bundle it was HANDED, as the union's
# active member. That bundle is still lazy, so the return's evidence lane found
# nothing to alias and materialized a second int. Only the second is declared
# owned, so the first was never released. str needs no materialization, which
# is exactly why it was already correct.
#
# The repair hands back the bundle whose values actually went into the lanes.
#
# ⛔ Why NOT teach the return to reuse the union's member lanes instead, which
# is where the second materialization is emitted: the return can only find the
# member lanes through the active-member record, so it would have to fix the
# same wrong record to read it. Recording the right bundle is the same repair
# one step earlier, and it also fixes every other reader of
# `unionActiveMember`.
#
# tests/golden/cases/string_annotation_union.py is in LYTHON_LEAK_GATE_CASES
# and pins this; it pins the VALUES too, which were correct throughout.
#
# differential: skip the leak is invisible to stdout; this records the shape


def pick(flag: bool) -> int | None:
    return 7


print(pick(True) is None)
