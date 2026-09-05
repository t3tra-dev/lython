# OPEN 2026-09-05. A `try` whose body YIELDS, inside a loop, inside a
# generator, is refused in the refcount phase:
#
#     error: unwind cleanup cannot target a handler entry with block arguments
#
# The generator state machine flattens the try and threads the loop's carried
# values through the handler entry's BLOCK ARGUMENTS -- (i64, i1) evidence pairs
# for `i` and the yielded value -- and `getOrCreateCleanupHandler` branches to
# that entry with no operands, so it refuses any handler that takes some.
#
# ⭐ THE CROSSINGS, all measured, that say where the line is:
#   - `try` with a yield in its body but NO loop around it: compiles.
#   - `try` AROUND the loop, yield inside the loop: compiles.
#   - `try` inside the loop with the yield inside it: this file.
#   - the same try/except in a loop OUTSIDE a generator: compiles.
#
# ⛔ `with` inside a generator is the same limit reached through the try the
# `with` lowers to, and it took a separate fix to get here: the yield-type walk
# did not bind a `with ... as X` target, so every such generator was refused
# EARLIER with "annotated Iterator[int] but yields builtins.object" -- a
# sentence about an annotation that was correct. `with Ctx(): yield 1`, with no
# target at all, reaches a third limit ("generator resume continuation live
# closure violated").
#
# ⛔ NOT the handler's operands taken from its existing branch: at the unwind
# site the loop's carried values are the CURRENT ones, not the end-of-body ones
# that branch passes, so forwarding them would hand the handler different values
# than the normal edge does. Whatever the fix is, it has to answer that.
from typing import Iterator


def guarded(n: int) -> Iterator[int]:
    for i in range(n):
        try:
            yield 10 // (i - 1)
        except ZeroDivisionError:
            yield -1


print(list(guarded(3)))
