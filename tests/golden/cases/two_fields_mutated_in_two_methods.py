# What this pins: a pool that moves values between two list fields.
#
#     p = Pool(); p.take(); p.give(3); print(p.free, p.used)
#     # error: cannot adapt builtins.list to runtime input 0 of
#     # builtins.list.__len__ [values:, expected 'memref<9xi64>']
#     # reported at `self.free.pop()`
#
# Nothing was wrong with the pop. The receiver bundle handed to the bound-method
# lowering was a REFERENCE into `valueBundles`, a DenseMap that same function
# inserts into on nearly every path; an insertion that rehashes moves the entry,
# and the liveness pin emitted after the pop then read freed memory and found a
# bundle with no physical values.
#
# So the trigger was how many bundles the program has, not the statement it was
# reported on. Every one of these made it compile again: dropping either
# statement of `give`, swapping their order, declaring `used` before `free`,
# initialising `used` to `[0]` instead of `[]`, or printing only ONE of the two
# fields. That is also why this file is kept SMALL and its shape exact: adding
# two more classes to it was enough to move it off the boundary and it stopped
# being able to fail at all.
#
# Why this needs to run rather than assert on a diagnostic: a dangling read of a
# DenseMap entry can also come back as a PLAUSIBLE bundle, which is a wrong
# answer rather than an error. What is pinned is which list holds which element
# after the moves.
#
# ⛔ This case cannot be made to cover the neighbours listed above -- putting
# them in the same file is what silenced it. They are recorded in
# tests/probe/wb_grid_leftovers_2026_08_16.py instead.
#
# Every expected line is python3.14's.


class Pool:
    def __init__(self) -> None:
        self.free: list[int] = [1, 2, 3]
        self.used: list[int] = []

    def take(self) -> int:
        v = self.free.pop()
        self.used.append(v)
        return v

    def give(self, v: int) -> None:
        self.used.remove(v)
        self.free.append(v)


p = Pool()
print(p.take(), p.free, p.used)
p.give(3)
print(p.free, p.used)
print(p.take(), p.free, p.used)
p.give(3)
print(p.free, p.used, len(p.free), len(p.used))


r = Pool()
i = 0
while i < 3:
    r.take()
    i += 1
print(r.free, r.used, len(r.used))
for v in [1, 2, 3]:
    r.give(v)
print(sorted(r.free), r.used)
