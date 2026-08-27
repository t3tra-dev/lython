# probe: leak -- an element read back out of a container an INNER loop builds
# from a name the OUTER loop bound (4000 iterations)
# axes: op=leak-loop iterations=4000
#
# The read-back mints a retain-rooted token on an object the outer loop's own
# value also holds, and its release is placed by liveness inside what the mint
# dominates. A release the placer dropped as "not dominated" and did not put
# back somewhere else is one leaked element per trip.
#
# ⛔ THE ELEMENT IS A LIST, not an int: a leaked small int is a refcount bump on
# a cached object and weighs nothing, so the probe would report clean over a
# real leak (see leak_optionalwalk_loop_small.py for the same instrument floor).
#
# CPython 3.14 expects: 4000


def run(n: int) -> int:
    total = 0
    a = 0
    while a < n:
        payload = [a, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6, a + 7]
        a = a + 1
        b = 0
        while b < 2:
            b = b + 1
            ys = [payload]
            total += len(ys[0]) - 7
    return total


print(run(4000) // 2)
