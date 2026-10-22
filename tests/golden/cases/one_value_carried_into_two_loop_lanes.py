# WHAT: loops that carry two locals bound to the SAME value on one edge --
# both rebound to the loop variable, one of them first bound to a parameter
# -- for ints, strs and ints past the machine word, run zero and several
# times.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the loop edge gave one
# token to two lanes and both released it, and the lane the parameter
# entered got no retain -- "borrowed entry argument 0 ... released or
# transferred without a prior retain". The refcounts only balance if every
# trip leaves both locals holding live values, which the values read after
# the loop show.
def ints(start: int, n: int) -> int:
    a, b = 0, start
    for j in range(n):
        a, b = j, j
    return a + b


def strs(n: int) -> list[str]:
    a, b = "x", "y"
    out: list[str] = []
    for j in range(n):
        s = str(j) * 3
        a, b = s, s
        out.append(a + b)
    return out + [a, b]


def bigs(n: int) -> int:
    big = 10 ** 30
    m, k = big, big
    for j in range(n):
        m, k = j + big, j + big
    return m + k - 2 * big


def separate(start: int, n: int) -> int:
    first = 0
    second = start
    for j in range(n):
        first = j
        second = j
    return first + second


print(ints(7, 3), ints(7, 0), strs(3), strs(0), bigs(3), separate(5, 4), separate(5, 0))
