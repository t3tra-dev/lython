# What: a `bool` local that is live across a `yield`. The generator's frame
# lanes are keyed on a runtime contract and a bool has one, but the gate asked
# with bools excluded -- for a reason that had stopped being true: the frame
# WORD accounting already gives a bool lane one word and the STORE side already
# takes one word for a bare i1. Only the LOAD half was never written, so the
# whole family compiled to "a value of type builtins.bool is live across a
# yield and has no generator frame lane".
#
# Running it is the only evidence the bit SURVIVED the suspension: the flag is
# read after the resume, and a lane that stored two zero words and rebuilt a
# memref from them would not come back True.
from typing import Iterator


def alternating() -> Iterator[bool]:
    flag = True
    for _ in range(4):
        yield flag
        flag = not flag


def seen_a_big_one(xs: list[int]) -> Iterator[int]:
    seen = False
    for x in xs:
        yield x
        if x > 1:
            seen = True
    yield 100 if seen else 0


def two_flags(a: bool, b: bool) -> Iterator[int]:
    yield 0
    yield (1 if a else 0) + (2 if b else 0)


def beside_an_int() -> Iterator[int]:
    n = 5
    flag = True
    yield 0
    yield n if flag else -n


print("alternating", list(alternating()))
print("loop flag", list(seen_a_big_one([1, 2, 3])), list(seen_a_big_one([0])))
print("parameters", list(two_flags(True, False)), list(two_flags(False, True)))
print("beside an int", list(beside_an_int()))
