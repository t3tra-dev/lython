# WHAT: an element read back out of a container the INNER loop builds from a
# name the OUTER loop bound, at four shapes -- a list, a tuple, one element and
# two, and the accumulator that reads them.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the sums are the assertion.
# The token under test is an increment on an object something else also holds,
# so getting its release wrong reads as a leak or a double free rather than as a
# wrong answer -- and the leak probe beside this one measures the first while
# these totals catch the second.
#
# ⛔ THE COPY `i = a` IS LOAD-BEARING. Without it the element's source is the
# loop header's argument directly, no owned-local marker is minted for the read
# back, and none of this is exercised: `m_noicopy` in the reduction set compiles
# on either side of the repair.


def lists(n: int) -> int:
    total = 0
    a = 0
    while a < n:
        i = a
        a = a + 1
        b = 0
        while b < n:
            j = b
            b = b + 1
            ys = [i, j]
            total += ys[0] + ys[1]
    return total


def one_element(n: int) -> int:
    total = 0
    a = 0
    while a < n:
        i = a
        a = a + 1
        b = 0
        while b < n:
            b = b + 1
            zs = [i]
            total += zs[0]
    return total


def tuples(n: int) -> int:
    total = 0
    a = 0
    while a < n:
        i = a
        a = a + 1
        b = 0
        while b < n:
            b = b + 1
            ts = (i,)
            total += ts[0]
    return total


def strings(n: int) -> int:
    total = 0
    a = 0
    while a < n:
        s = "k" + str(a)
        a = a + 1
        b = 0
        while b < n:
            b = b + 1
            ws = [s]
            total += len(ws[0])
    return total


def over_range(n: int) -> int:
    total = 0
    for a in range(n):
        i = a
        for b in range(n):
            ys = [i, b]
            total += ys[0] + ys[1]
    return total


print(lists(4), one_element(4), tuples(4), strings(4), over_range(4))
print(lists(1), one_element(1), tuples(1), strings(1), over_range(1))
print(lists(7), one_element(7), tuples(7), strings(7), over_range(7))
