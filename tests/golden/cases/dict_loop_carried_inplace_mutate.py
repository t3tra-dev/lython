# A dict carried through two consecutive loop headers where the SECOND loop both
# iterates it and mutates it in place, and completes normally.
#
# What this pins that the other dict goldens do not. The token moves from the
# producer to the first header's argument, and from the first loop's EXIT block
# to the second header's argument -- a transfer at a branch, two links deep, in a
# block that neither produces the value nor is the destination. Every other
# loop-carried dict case in this suite either raises inside the second loop
# (dict_changed_size, dict_iteration_views) or never mutates in it, so the
# completing path was never executed: an owner group missing for the second link
# is a compile-time diagnostic, but an owner group present with its borrow-edge
# retain silently dropped is a runtime over-release, and only a loop that runs to
# the end reaches the release.
#
# Mutating an EXISTING key is load-bearing: a new key changes the size and the
# iteration guard raises, which is the path the other cases already cover.
d: dict[int, int] = {}
for i in range(5):
    d[i] = i

acc = 0
for k in d:
    d[0] = k
    acc = acc + k
print(acc)
print(len(d), d[0], d[4])

# Three links, so the chain has to keep going past the second.
first = 0
for k2 in d:
    d[1] = k2
    first = first + 1
total = 0
for v in d.values():
    total = total + v
print(first, total)
