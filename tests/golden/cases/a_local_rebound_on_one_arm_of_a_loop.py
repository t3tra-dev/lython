# `if <cond>: <outer local> = <loop-carried value>` -- the index-of-the-best
# loop, and one of the plainest shapes Python has. It did not COMPILE: the
# conditional rebind lowers to a select over two loop-carried block arguments,
# and the loser's reference was never released, so the retain that pairs with
# the branch's read of the select had no discharge and the affine ownership
# verifier refused the program ("released owned resource from @LyLong_FromI64
# is used after release").
#
# Golden and not a driver assertion because the values are the other half: the
# repair puts the select back into a branch, and a branch that forwards the
# wrong arm computes a wrong best. The million-range arm is deliberate -- small
# ints are shared, so a refcount error on them is invisible; these are heap
# boxes. Registered in the leak gate too, which is what watches the release
# this repair supplies.
best_index = 0
for i in range(1, 6):
    if i % 2 == 0:
        best_index = i
print(best_index)

values = [3, 1, 4, 1, 5, 9, 2, 6]
best = 0
for index in range(1, len(values)):
    if values[index] > values[best]:
        best = index
print(best, values[best])

big = 0
for n in range(1000000, 1000060):
    if n % 7 == 0:
        big = n
print(big)

word = ""
for candidate in ["pear", "banana", "fig", "clementine"]:
    if len(candidate) > len(word):
        word = candidate
print(word)


def argmin(xs: list[int]) -> int:
    at = 0
    j = 1
    while j < len(xs):
        if xs[j] < xs[at]:
            at = j
        j += 1
    return at


print(argmin(values), values[argmin(values)])
