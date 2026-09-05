# OPEN 2026-09-05. A short-circuit `and`/`or` in the condition guarding a YIELD,
# inside a generator's loop, loses the unwind release for a list local:
#
#   error: owned resource from @LyList_FromLength result 0 is still owned when a
#   call to 'LyLong_FromI64' may unwind out of the function; the unwind path
#   must release, transfer, or return it
#
# ⭐ THE CROSSINGS, all measured, and the last one is the shape of the repair:
#   - the same condition written as NESTED `if`s: compiles.
#   - `if buf:` alone (one operand): compiles.
#   - `or` in place of `and`: fails the same way.
#   - dropping the `buf = []` after the yield: still fails, so it is the guard
#     and not the replacement.
#   - the same function returning a LIST instead of yielding: compiles.
#
# A short-circuit operand lowers to its own blocks, and the generator's state
# machine splits the loop at the yield inside them -- so the list's owned token
# is live on an edge the cleanup placement does not cover. Nested `if`s produce
# the same control flow to a reader and different blocks to the pass, which is
# what says the gap is in the placement rather than in the program.
#
# ⛔ Related to wb_a_try_inside_a_loop_inside_a_generator.py: both are the
# refcount phase failing to place a release on an exceptional edge inside a
# generator, and both work outside one.
from typing import Iterator


def blocks(lines: list[str]) -> Iterator[tuple[str, list[str]]]:
    kind = "para"
    buf: list[str] = []
    for line in lines:
        if line.startswith("-"):
            if kind != "list" and buf:
                yield (kind, buf)
                buf = []
            kind = "list"
        buf.append(line)
    if buf:
        yield (kind, buf)


print(list(blocks(["a", "-b"])))
