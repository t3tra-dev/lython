# What this pins: a generator that iterates range() and has a branch in its body
# -- the filtered generator, which is most of them.
#
#     def evens(n: int):
#         for i in range(n):
#             if i % 2 == 0:
#                 yield i
#     # owned resource from @LyRange_Iter result 0 is still owned when a call to
#     # 'LyLong_FromI64' may unwind out of the function
#
# The refusal was RIGHT about the leak and wrong only about there being nothing
# to do. The call it named is the boxing of the yielded value in the suspend
# block, and the iterator is transferred out by that block's return -- so an
# unwind out of the boxing call left the frame with nobody holding the iterator.
# No cleanup could be placed because the suspend block's argument had no tracked
# group: group collection maps a forward to its destination and BAILS when a
# terminator hands the same values to two successors, which is exactly what a
# suspend is (`cond_br %susp, ^suspend(%it), ^loop(%it)`). The release side is
# right to bail -- which destination would own it? -- but on an unwind both
# destinations hold the token, so the unwind list follows each successor.
#
# Why this must run: the shapes below differ only in where the branch puts the
# yield, and each one is a different block for the boxing call to land in. What
# says the repair works is that they all produce CPython's values -- and the
# `raise` case is what says the cleanup it adds is on the right edge, since that
# path unwinds out of a suspended generator and the leak gate measures it.
#
# ⛔ The while-loop spelling was the workaround and stays correct; it is here so
# a future change that "fixes" one and breaks the other is visible.
#
# `refilled` is a second defect the same programs walk into: an empty container
# literal has no element type of its own, so a rebind with one keeps the type the
# name already has -- the rule the emitter applies outside a generator, which the
# generator's own frame analysis did not. Overwriting it made the frame slot a
# union of the two readings ("runtime bundle for '!py.union<list[int],
# list[object]>' has 1 values"), which is the accumulate-and-flush idiom.
#
# ⛔ The same idiom INSIDE A LOOP (append, yield when full, rebind, continue) is
# still refused, now for an ownership reason rather than a typing one, and is
# recorded in tests/probe/wb_sweep_findings_2026_08_18.py.


def evens(n: int):
    for i in range(n):
        if i % 2 == 0:
            yield i


def scaled(n: int):
    for i in range(n):
        if i % 2 == 0:
            yield i * 10
        else:
            yield -i


def skipping(n: int):
    for i in range(n):
        if i % 2 == 0:
            continue
        yield i


def while_form(n: int):
    i = 0
    while i < n:
        if i % 2 == 0:
            yield i
        i += 1


def refilled():
    buf: list[int] = []
    buf.append(1)
    yield buf
    buf = []
    yield buf


def consume(n: int) -> int:
    total = 0
    for v in evens(n):
        total += v
        if total > 4:
            raise ValueError("stop")
    return total


print(list(evens(6)), list(scaled(4)))
print(list(skipping(4)), list(while_form(4)))
print(list(refilled()))
for v in evens(6):
    print(v)
try:
    print(consume(10))
except ValueError as e:
    print("caught", e)
