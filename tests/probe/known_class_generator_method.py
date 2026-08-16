# probe: REPORTED loud: a generator method on a class
# CLASSIFICATION @ 2026-08-17: RUNS (6)
#
# ⭐ FIXED, in three steps that are each worth keeping. The signature was
# computed before the class contracts existed (yield type `object`); the `for`
# loop kept its position in a cell, which cannot survive a suspension; and the
# receiver arrived as a CLOSURE CAPTURE.
#
# ⛔ THE THIRD ONE'S DIAGNOSTIC WAS WRONG, and four measurements said so: an
# EMPTY class refused too, reading the field into a local before the first
# yield refused, self.xs likewise, and the identical body as a module-level
# function ran. The message blamed the layout ("only builtins.int and manifest
# contracts with a rank-1 physical shape have a resume lane, and a user class
# has neither") -- but `generatorLaneParts` computes that layout for a source
# class from its ClassOp, and the state machine builds its argument lanes from
# the callable's POSITIONALS, which the bound form has none of. Its own advice
# ("read the value into an int local before the first yield") did not work
# either.
#
# ⛔ GIVING THE CAPTURE A LANE WAS TRIED AND REVERTED: the state machine then
# accepts (pos=0 clos=1 argEligible=1 lanes=1, livesEligible=1) and the
# identical message comes from the resume clone, whose frame-lane lookup is
# `inputIndex - generatorControlCount` and never consults argument lanes. The
# fix is one layer up instead -- a DIRECT call takes the method SYMBOL with the
# receiver as the leading positional, the route a recursive method already
# took, and as a positional the receiver rides the lane that always existed.
#
# ⛔ `m = b.each` then `m()` still builds the bound object and still refuses:
# there is no call to attach the receiver to. Same for a keyword argument,
# which would need the recursive path's slot placement. Both are recorded in
# tests/golden/cases/generator_method_receiver_lane.py.
#
# CPython 3.14 expects: 6

from typing import Iterator


class Bag:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs

    def each(self) -> Iterator[int]:
        for x in self.xs:
            yield x


b = Bag([1, 2, 3])
total = 0
for v in b.each():
    total += v
print(total)
