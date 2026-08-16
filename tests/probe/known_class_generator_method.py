# probe: REPORTED loud: a generator method on a class
# axes: op=generator-method flow=for
# CLASSIFICATION @ 2026-08-17: 3 loud 拒否 (診断)
#   source generator next lowering currently supports yields whose runtime
#   value is a single lane, and '!py.contract<"builtins.int">' has 3
#
# ⛔ RECLASSIFIED. It used to be "generator function return annotation is
# incompatible with inferred Generator or AsyncGenerator contract", which was
# the SIGNATURE being computed before the class contracts existed -- fixed
# 2026-08-17, and the probe moved one layer down to the real blocker. The yield
# type is now `int`; what stops it is that an int read out of a LIST is the
# 3-lane object form, while the generator frame carries a single lane. That is
# the int-only yield plan recorded on the seven-gap cluster, and
# `for i in range(n): yield i` works because a range element rides an i64 lane.
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
# by contract name today) would have to key on the logical one. That is one
# mechanism; it was scoped, not built.
#
# ⭐ AND THERE IS A SECOND ONE, FOUND 2026-08-16, which does not touch the lane
# at all: GIVE THE LIST A REAL ITERATOR CONTRACT. The reason `range` works is
# not that ranges are special -- it is that `builtins.range_iterator` is a
# manifest class with `ly.runtime.shape`, `alloc`, `__iter__`, `__next__` and a
# deallocator. `builtins.str_iterator` is the same pattern, and `for c in s:
# yield c` inside a generator gets past this scan for exactly that reason (it
# then fails further downstream, in `str.join`). Those are the only two
# iterator contracts the manifest has:
#
#     ly.runtime.contracts = [..., "builtins.range_iterator",
#                                  "builtins.str_iterator", ...]
#
# A list is iterated as an index walk over the container instead, so `py.iter`
# has nothing concrete to answer with and produces the protocol. A
# `builtins.list_iterator` (header + index + the list handle, mirroring
# str_iterator's header + state + source) would make this generator eligible
# with no change to the lane machinery, and would carry tuple/dict/set behind
# it. Which of the two mechanisms is right is a design call: the lane one is
# general and touches the suspension ABI, this one is local and adds a runtime
# class per container.

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
