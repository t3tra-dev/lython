# One source filling TWO slots of a literal, then a delete. Execution is needed
# because the defect was a MISSING release and the values were right without it:
# the frame's reference outlived the frame, so nothing printed differently and
# only the allocator could see it. This file exists for
# LYTHON_LEAK_GATE_CASES; the printed lines are what keeps it honest about the
# list still being correct after the delete.
#
# The three spellings around it are here because none of them showed the defect
# and a repair that reroutes them would go unnoticed otherwise: one slot from
# the same source, two slots from distinct sources, and two slots with no
# delete at all.


def duplicated() -> None:
    msg = "hi there"
    xs = [msg, msg]
    del xs[0]
    print(len(xs), xs[0])


def single() -> None:
    msg = "solo"
    xs = [msg]
    del xs[0]
    print(len(xs))


def distinct() -> None:
    xs = ["p", "q"]
    del xs[0]
    print(len(xs), xs[0])


def no_delete() -> None:
    msg = "kept"
    xs = [msg, msg]
    print(len(xs), xs[0], xs[1])


duplicated()
single()
distinct()
no_delete()
