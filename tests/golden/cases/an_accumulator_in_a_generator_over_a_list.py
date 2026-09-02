# What: a generator whose loop-carried int comes out of a CONTAINER. The frame
# carries an int as a box plus an unboxed word and the resume entry forwards
# only the word into the body, so a value that never had one -- an int read out
# of a list is a boxed object with no lane -- came back invalid and the next
# suspend raised "int too large to convert to a native 64-bit integer" for the
# int 1.
#
# WHY THIS IS RUN: the value is wrong only after a SUSPENSION, so the first
# element is right whatever the frame does and the second is the whole test.
# The decode is that the accumulated numbers are printed rather than counted: a
# frame that loses the word raises, and one that carries a stale word prints a
# plausible sequence with the wrong sums in it. The `range` spelling below is
# the control -- its counter has a word all along.
def running(xs: "list[int]"):
    total = 0
    for x in xs:
        total = total + x
        yield total


print([v for v in running([1, 2, 3])])


def counted(n: int):
    total = 0
    for x in range(n):
        total = total + x
        yield total


print([v for v in counted(4)])


def last_of(xs: "tuple[int, int, int]"):
    last = 0
    for x in xs:
        last = x
        yield last * 10


print([v for v in last_of((4, 5, 6))])


def indexed(xs: "list[int]"):
    seen = 0
    i = 0
    while i < len(xs):
        seen = xs[i]
        yield seen
        i = i + 1


print([v for v in indexed([7, 8])])


# The same value as the generator's ARGUMENT rather than its carried local: the
# argument lane is the word alone, with no box beside it, so this one did not
# compile at all while `doubled(len(xs))` did.
def doubled(k: int):
    yield k
    yield k + 1


def from_first(xs: "list[int]") -> "list[int]":
    return [v for v in doubled(xs[0])]


print(from_first([5, 9]))
