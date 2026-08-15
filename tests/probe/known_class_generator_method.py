# probe: REPORTED loud: a generator method on a class
# axes: op=generator-method flow=for
# CLASSIFICATION @ kernel/4b fa71a3c: 3 loud 拒否 (診断)
#   generator function return annotation is incompatible with inferred Generator or AsyncGenerator contract
# CPython 3.14 expects: 6
#
# ⭐ LOCALIZED 2026-08-15, and it is NOT the method, the class, or the loop.
# It is the ITERATOR'S TYPE. Measured, all three in one session:
#
#   for i in range(n): yield i .................... runs  (golden
#                                                   generator_for_loop)
#   i = 0; while i < len(xs): yield xs[i]; i += 1 . runs
#   for x in xs: yield x    (xs: list[int]) ....... refused  <- this shape
#
# The range form's iterator is `!py.contract<"builtins.range_iterator">`, a
# concrete manifest contract with a lane. A list's is
# `!py.protocol<"Iterator", [!py.contract<"builtins.int">]>`, which is
# object-erased: `runtimeContractName` answers "" for it, so the state
# machine's frame-lane scan (`laneEligibleContract`, GeneratorStateMachine.cpp)
# declares the whole generator ineligible and it falls back to the inline
# tier, whose refusal is the message printed today. The iterator is live across
# the yield, which is the only reason it needs a lane at all.
#
# ⛔ Why NOT just name the lane "builtins.object", which is what
# `runtimeShapeContractName` already answers for a protocol and what the rest
# of the ABI does with erased values: the frame lane's contract also types the
# clone's block ARGUMENT (`runtimeContractType(context, lane.contract)`), and
# the continuation's use is `py.next @__next__ : ...(%it : !py.protocol<...>)`.
# An object-typed argument does not spell that operand. The lane would have to
# carry a LOGICAL type beside its physical shape, and the lane grouping (keyed
# by contract name today) would have to key on the logical one. That is the
# mechanism; it was scoped, not built.

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
