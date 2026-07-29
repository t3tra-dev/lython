# Reading the SAME container slot more than once, where the element is itself a
# container.
#
# Why this needs to execute: the defect it pins is a LEAK, invisible to exit code
# and stdout, and the `leak` stage (tests/leak_gate.py) takes a golden case as its
# input. The stdout below is a bonus; the assertion is net-zero allocation at exit.
#
# What it pins. Ownership of a frame-local is tracked PER SSA VALUE downstream, so
# `refcount-insertion` emits one release for a value it finds marked, however many
# times it is marked. Two reads of one slot reconstruct the SAME handle, so the
# lowering used to mint two owned tokens on one value: two retains, one release,
# and the whole inner entity -- handle, items array and every box it owns -- was
# never freed. It saturated rather than scaling with the reads, which is what a
# per-value map looks like from the outside, and it did not depend on the outer
# container's length.
#
# The invariant is one owned token per SSA value. It was assumed by the consumer
# and enforced nowhere; the producer now borrows an existing token instead of
# minting a second, which is also what a re-read MEANS -- the frame already holds
# a reference, so the second name is a borrow and a borrow costs nothing.
#
# Sizes, so a regression is legible rather than just red: a two-element inner
# tuple leaked 2 roots / 10368 B, a seventy-element one leaked 69 / 14656 B.
# The seventy-element read below is therefore the loudest line here.

n = ((1, 2), (3, 4))
a = n[0]
b = n[0]
c = n[0]
print(a, b, c)

# Two slots, each read twice: the leak was per (container, slot), so this was
# twice the above rather than once.
d = n[0]
e = n[1]
f = n[0]
g = n[1]
print(d, e, f, g)

# A list of tuples: the container's contract does not matter, the element's does.
xs = [(1, 2), (3, 4)]
h = xs[0]
i = xs[0]
print(h, i)

# A variable index rather than a literal one, so the reread is not something a
# constant-folder could have collapsed before the ownership machinery saw it.
k = 1
m = n[k]
o = n[k]
print(m, o)

# The seventy-element inner tuple, whose leak was 69 roots.
big = ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
        38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69), (1, 2))
p = big[0]
q = big[0]
print(len(p), len(q), p[69], q[0])
