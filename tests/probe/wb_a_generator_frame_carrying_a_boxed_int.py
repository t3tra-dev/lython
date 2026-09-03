# OPEN, and now LOUD instead of silent. A generator that binds a dispatched
# int to a local and yields it:
#
#     def g(items: list[Shape]) -> Iterator[int]:
#         for s in items:
#             v = s.area()
#             if v > 0:
#                 yield v
#
# raises "ValueError: int too large to convert to a native 64-bit integer".
# The frame carries a local across a suspension in an i64 lane, and the
# dispatcher's result is a boxed int with no word to put there.
#
# ⭐ IT USED TO ANSWER `[]` INSTEAD -- silently wrong. The resume is compiled as
# a primitive-i64 clone, and a comparison whose operands cannot be vouched for
# parked "I cannot say" and answered FALSE, which the resume then branched on.
# That is sound only where the CALLER re-runs the boxed original, and a
# resume's caller is the runtime's `next`. The comparison now takes the guarded
# path in a resume (cases/a_generator_that_branches_on_a_dispatched_int), which
# is what turned this shape from a wrong answer into the frame's own refusal.
#
# ⛔ WHAT IS LEFT is the frame, not the comparison: an int the frame must carry
# has to have a word, and a boxed one does not. The same three shapes
# [[lython-unboxed-int-lane]] records. The spelling that does NOT bind a local
# (`yield s.area()`, or `if s.area() > 0: yield s.area()`) works, because
# nothing crosses the suspension.
#
# Measured 2026-09-04.
from typing import Iterator


class Shape:
    def area(self) -> int:
        return 0


class Sq(Shape):
    def __init__(self, n: int) -> None:
        self.n = n

    def area(self) -> int:
        return self.n * self.n


def g(items: list[Shape]) -> Iterator[int]:
    for s in items:
        v = s.area()
        if v > 0:
            yield v


shapes: list[Shape] = [Shape(), Sq(3)]
print(list(g(shapes)))
