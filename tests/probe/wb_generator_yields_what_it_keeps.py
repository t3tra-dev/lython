# FIXED 2026-08-30, in two repairs, and this program needed both.
#
# It yields a list it also keeps binding, so the value has TWO holders at every
# suspend -- the consumer and the frame -- and it aborted in `Ly_DecRef` on the
# second trip.
#
# ⭐ THE LANE COMPARISON WAS ASKING ABOUT THE WRONG THING.
# `appendGeneratorLaneReturnOperands` already retains when two lanes carry one
# value; the caller (ABI/Returns.cpp) deduped on the LOGICAL return operand,
# and `current.append(v)` rebinds it, so the yield lane held the call result
# and the frame lane the block argument the loop threads. Comparing the
# physical group fixes that; comparing what every predecessor forwards into
# each lane's block argument fixes the conditional yield, where the two lanes
# arrive as arguments 1 and 9 of one suspend block. `resolveLaneEntity` is
# both steps.
#
# ⭐ AND ONE TOKEN CANNOT BE TRANSFERRED TWICE ON ONE EDGE. With the lanes
# fixed the abort moved to `Ly_IncRef` on a dead object: the loop's merge takes
# both the value it carries on and the value it releases as "replaced", and on
# the trip that did not rebind they are the same object. Each argument group
# asked `diesOnEdge` alone and each said yes. The same loop outside a generator
# is sound only because there the release names the pre-merge value directly --
# a use past the edge -- which is why "the same loop, not in a generator" read
# as clean and made this look like a generator-only defect.
#
# The earlier note recorded two keys that each fixed one shape and left the
# next; the boundary they could not characterise was the second repair, not a
# third key. THE NEIGHBOURS, all measured green now:
#   yield without a rebind, rebind without the append, the same loop outside a
#   generator, `for` over a list parameter, `list(chunks(...))`, a
#   comprehension, two generators of this shape in one module.
#
# ⛔ A yield inside a `try` in a loop is still refused -- "unwind cleanup cannot
# target a handler entry with block arguments" -- which is a different
# mechanism and not a mis-execution.
#
# Goldens: cases/a_generator_yields_the_list_it_keeps_filling (first repair),
# cases/a_generator_replaces_the_list_it_just_yielded (second).
def chunks(values: list[int], size: int):
    current: list[int] = []
    for v in values:
        current.append(v)
        if len(current) == size:
            yield current
            current = []


for chunk in chunks([1, 2, 3, 4], 2):
    print(chunk)
