# OPEN, and the shape is the whole pipeline idiom. A generator VALUE that
# crosses a call boundary cannot be iterated on the other side:
#
#     def numbers(n: int) -> Iterator[int]:
#         i = 0
#         while i < n:
#             yield i
#             i += 1
#
#     def total(source: Iterator[int]) -> int:      # a plain function!
#         out = 0
#         for v in source:
#             out += v
#         return out
#     total(numbers(4))
#     # a generator returned out of a function cannot be resumed here: the
#     # frame it resumes into is not reachable from this read
#
# MEASURED 2026-09-08 (RelWithDebInfo). Three spellings, three messages, one
# missing thing:
#
#   for v in source:  in a plain function taking the generator ..... the above
#   for v in numbers(6):  inside another GENERATOR .................. the above
#   yield from source:  in a generator taking the generator ......... "source
#       generator next lowering currently supports only straight-line pure int
#       yield bodies"
#   for v in numbers(6):  inside a plain function ................... correct
#   for v in source:  where source is a `list[int]` parameter ....... correct
#   `for v in list(inner())` (the advice the message gives) ......... correct
#
# ⭐ WHAT IS MISSING IS THE TARGET, NOT THE VALUE. A generator's resume is a
# STATIC call to the body's resume function, chosen from the bundle's
# `generatorTarget` -- and a parameter's bundle is rebuilt from its TYPE,
# `types.GeneratorType`, which names no target. So the callee has the object
# and no idea whose frame it is.
#
# ⛔ SO THE REPAIR IS THE ARGUMENT SPECIALIZER, not the frame. `type[X]`
# already takes this road: a parameter whose type does not determine the
# callee's behaviour gets ONE BODY PER GROUND ARGUMENT, and which generator
# body a value resumes is exactly that kind of fact. `yield from` on such a
# parameter needs the same key before its own limit is even reachable.
#
# ⛔ NOT the frame-lane work (GeneratorStateMachine.cpp), which is about a
# value living ACROSS a suspension. These three fail with no suspension in
# sight -- `total` is not a generator at all.
from typing import Iterator


def numbers(n: int) -> Iterator[int]:
    i = 0
    while i < n:
        yield i
        i += 1


def total(source: Iterator[int]) -> int:
    out = 0
    for v in source:
        out += v
    return out


print(total(numbers(4)))
