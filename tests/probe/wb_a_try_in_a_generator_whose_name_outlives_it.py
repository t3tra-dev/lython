# probe: a `try` inside a GENERATOR whose body binds a name read AFTER the try
# CLASSIFICATION @ c2dbab5b: 5 loud, all from the ownership pass
#   "unwind cleanup cannot target a handler entry with block arguments"
# CPython 3.14 expects: [1]
#
# The trigger is exact, and each of these was measured:
#   try/except, name read after ................. refused
#   try/finally, name read after ................ refused
#   the same with a str instead of an int ....... refused
#   the name NOT read after the try ............. compiles
#   `if/else` instead of `try` .................. compiles
#   the `yield` INSIDE the try .................. compiles
#
# ⭐ THE MECHANISM. A generator's state machine threads its FRAME through BLOCK
# ARGUMENTS, so every block of a resumed body takes one. Reading the name after
# the try makes a join, the join makes the handler entry a block with
# arguments, and an unwind edge has no operands to pass -- which is the
# sentence above, from `insertUnwindCleanupReleases` in
# Passes/Runtime/Passes/Ownership.cpp.
#
# ⭐ AND THE ARGUMENT CARRIES NOTHING. Every predecessor passes the SAME value
# (the frame view, defined once in the allocation block), so replacing the
# argument with it and dropping it is semantics-preserving and makes all five
# shapes above compile and print CPython's answer. That was built and measured.
#
# ⛔ AND IT IS NOT SAFE WHERE IT HAS TO RUN, which is why it was dropped rather
# than shipped. Two measurements:
#
#   1. In the cleanup loop: the tracked ownership groups hold mlir::Values and
#      one of them can BE the argument being erased. The pass crashed
#      INTERMITTENTLY -- a use-after-free, so sometimes it did not. 1 in ~18
#      runs of the same program.
#   2. Moved to a pre-pass before the marker scan: still intermittent, because
#      `groups` is computed for the whole module BEFORE this step and holds the
#      same Values.
#   3. And with the try inside a `for`, the back edge's disagreeing operand is
#      not visible at this point, so the uniformity test passes and the drop
#      produces IR that VERIFIES and then double-frees at run time
#      ("Ly_DecRef observed non-positive refcount"). An acyclicity gate stops
#      that one, but not 1 or 2.
#
# The repair belongs BEFORE the ownership groups are built -- a canonicalization
# ahead of RefCountInsertion, not a fixup inside it.

from typing import Iterator


def go() -> Iterator[int]:
    v = 0
    try:
        v = 1
    except ValueError:
        v = -1
    yield v


print(list(go()))
