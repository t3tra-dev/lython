# FIXED 2026-08-14. Kept as the reproducer and the bisect.
#
# WAS: one loop element leaked per iteration -- 3 allocations / 156 B for a
# three-element list, 5 / 260 B for six.
#
# THE CAUSE, and it was one line from the machinery that already existed: the
# select-to-diamond expansion (Passes/Ownership.cpp) only fires when an
# operand is frame-owned, and `frameProduces` asked that question of
# `underlyingObjectValue(value)` -- which reads THROUGH the
# `owned_local_object` marker cast to the raw alloc, whose defining op
# carries no attribute. So the one shape the expansion exists for answered
# "not frame-owned". Asking the value itself first, before peeling, is the
# repair; net 0 on every bisect line below.
#
# BISECTED (tests/leak_gate.py against ./build/bin/lyc):
#
#   if value < lo:  lo = value    ... 3 allocs / 156 B   <- this file
#   the same with a str list ......  3 allocs / 123 B
#   if value < 5:   lo = value ....  3 allocs / 156 B   (the condition is
#                                    not the trigger)
#   for value in xs (a parameter) .  3 allocs / 156 B   (nor the literal)
#   if value < lo:  lo = 0 ........  0 (clean)
#   lo = value, UNCONDITIONALLY ...  0 (clean)
#   if value < 0:   n = n + 1 .....  0 (clean)
#
# So the shape is exactly: a loop-carried local CONDITIONALLY rebound to the
# loop ELEMENT.
#
# LOCATED, in the IR the refcount passes leave
# (LYTHON_IR_DUMP=refcount-elision). The loop header materialises the element
# as an owned temporary, and the body merges it with the carried value
# through a `select`:
#
#     ^bb5(%60, %61, %62):                       ; the carried `lo`
#       call @Ly_IncRef(%cast_94)                ; the element, +1
#       %135:3 = ... {ly.ownership.owned_local_object}
#     ^bb8:
#       %139 = call @LyLong_LtBool(%135..., %60...)
#       %140 = arith.select %139, %135#0, %60    ; the next `lo`
#       call @Ly_IncRef(%cast_99) {block-arg-merge-borrow}
#       call @LyLong_DecRef(%60) {..:py.decref}  ; the old `lo`, -1
#       cf.br ^bb5(%140, %141, %142)
#
# A select-merge has TWO losers -- the carried value on the arm that rebinds,
# and the element on the arm that does not -- and only one release is placed.
# `%60` gets its decref; `%135` never does, on either arm. The borrow retain
# is not the error: it takes the merge argument's own token, which is what
# lets both losers be released.
#
# ⛔ Why NOT credit the element's retain as the edge's transfer instead, which
# is what the note at `emitterLaneIncrefInBlock` (Passes/Ownership.cpp) points
# at -- it looks for a retain labelled `":py.incref"` and this one carries no
# label at all: a transfer is only correct on the arm the select TAKES. On the
# other arm `%140` is `%60`, which the same block releases, so the merge would
# receive an object that has just been dropped. The missing half is the
# loser's release, not the winner's retain.
#
# NOT the forwarding-block defect fixed 2026-08-14
# (wb_generator_nested_loop_iterator_leak.py): measured identical, 8 roots /
# 383 B, on the binaries either side of that repair.
#
# tests/golden/cases/loop_conditional_rebind.py pins the VALUES, which are
# correct; nothing pins the leak, which is why this file exists.
#
# differential: skip the leak is invisible to stdout; this records the shape


def f() -> None:
    lo: int = 0
    for value in [4, -2, 9]:
        if value < lo:
            lo = value
    print(lo)


f()
