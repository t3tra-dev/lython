# What this pins: `for x in G(): yield x` inside a generator.
#
#     def relay() -> Iterator[int]:
#         for x in src():
#             yield x
#     # a generator returned out of a function cannot be resumed: ... Call the
#     # generator in the for statement, bind it to a local in the same function,
#     # or return a list
#
# The advice was already followed -- the generator IS called in the for statement.
# The same loop in a plain function and at module scope runs, and `yield from
# src()` inside a generator runs, so what had no path is a nested generator
# resumed across the OUTER generator's own suspensions.
#
# When the body is exactly one bare `yield` of the loop target, delegation is what
# the program means, and delegation has a path: the loop is written as
# `yield from`.
#
# Why this needs to run rather than assert on a diagnostic: delegation decides the
# ORDER and the COUNT of what the outer generator yields, and a rewrite that
# dropped or repeated an element compiles. Every section below consumes the whole
# generator; the two-loop section is there because each loop must delegate to its
# own fresh sub-generator.
#
# ⛔ A body that is anything else stays refused: `yield x * 2` is not delegation
# and `py.yield.from` cannot carry it. That is the nested-generator frame work
# the resume-target rule waits on.
#
# ⭐ THE DIAGNOSTIC'S ADVICE WORKS, and the note here used to say it did not.
# Re-measured 2026-09-07: `for x in list(src(n))` INSIDE a generator compiles
# and runs, so does binding the list to a local first, and so does a
# comprehension over it. The last three sections pin all three, because a note
# claiming the suggested spelling is refused is worse than no note -- it sends
# a reader looking for a workaround they already have.
#
# Every expected line is python3.14's.

from typing import Iterator


def src(n: int) -> Iterator[int]:
    i = 0
    while i < n:
        yield i
        i += 1


def relay(n: int) -> Iterator[int]:
    for x in src(n):
        yield x


def twice(n: int) -> Iterator[int]:
    for x in src(n):
        yield x
    for y in src(n):
        yield y


def doubled(xs: list[int]) -> Iterator[int]:
    for x in xs:
        yield x * 2


def doubled_inline(n: int) -> Iterator[int]:
    # The materialization the diagnostic suggests, INSIDE the generator.
    for x in list(src(n)):
        yield x * 2


def doubled_via_local(n: int) -> Iterator[int]:
    xs = list(src(n))
    for x in xs:
        yield x * 2


def doubled_via_comprehension(n: int) -> Iterator[int]:
    for x in [v * 2 for v in src(n)]:
        yield x


# --- the delegating loop, consumed whole -----------------------------------
print(list(relay(0)), list(relay(1)), list(relay(3)))
print(list(twice(2)))

# --- driven by a for statement, and by the aggregates ---------------------
total = 0
for v in relay(4):
    total += v
print(total, sum(relay(4)), max(relay(4)), len(list(relay(4))))

# --- a body that is not delegation, through each workaround ---------------
print(list(doubled(list(relay(3)))))
print(list(doubled_inline(3)), list(doubled_via_local(3)))
print(list(doubled_via_comprehension(3)), list(doubled_inline(0)))
