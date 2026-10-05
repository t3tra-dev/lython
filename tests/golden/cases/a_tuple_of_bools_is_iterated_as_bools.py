# WHAT: a for loop over a tuple of bools hands each element on as a bool --
# printed, negated, counted, and passed to a generator that branches on it.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: a tuple keeps its bools
# boxed, and each read turns the box back into a truth value at run time;
# DriverTest.IteratingATupleOfBoolsCompiles only says the loop compiles, not
# that the truth values it reads are the ones stored.
from typing import Generator


def gen(flag: bool) -> Generator[int, None, int | str]:
    yield 1
    if flag:
        return 5
    return "five"


for f in (True, False):
    print(f, not f)
    g = gen(f)
    next(g)
    try:
        next(g)
    except StopIteration as e:
        print(repr(e))
flags = (False, True, True)
n = 0
for f in flags:
    if f:
        n += 1
print(n, [not f for f in flags], any(flags), all(flags))
