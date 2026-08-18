# What this pins: `yield from` over something that is not a list literal.
#
#     def g() -> Iterator[int]:
#         yield from range(2)
#     # source generator next lowering currently supports yields whose ...
#
# A range, a parameter's list and a str all landed there, while the loop spelling
# of each -- `for x in xs: yield x` -- has always worked. So the gap was
# `py.yield.from` in the state machine and not the iteration, and `yield from X`
# is written as that loop now. The literal arm is unchanged: a list or tuple
# literal unrolls into one yield per element and needs no loop.
#
# Exact for an ITERABLE, which is what these operands are: `yield from` over one
# evaluates to None, and that is what the loop leaves behind.
#
# ⛔ NOT for a sub-GENERATOR. `py.yield.from` is what the state machine
# implements for delegation -- send and throw pass through it -- and rewriting
# those to a loop took two passing goldens down with "a generator returned out of
# a function cannot be resumed", because the loop iterates a generator VALUE,
# which is a different and separately refused shape. The rewrite is gated on the
# operand's type, and `generator_yield_from` is the case that guards it.
#
# Why this needs to run rather than assert on a diagnostic: the rewrite decides
# the ORDER and COUNT of what a generator yields. A loop that ran once too often,
# or that yielded the iterable instead of its elements, compiles -- so every
# section below consumes the whole generator and prints what came out.
#
# Every expected line is python3.14's.

from typing import Iterator


def counted(n: int) -> Iterator[int]:
    yield from range(n)


def mixed(n: int) -> Iterator[int]:
    yield from range(n)
    yield from [100, 200]
    yield 7


def from_parameter(xs: list[int]) -> Iterator[int]:
    yield from xs


def from_str(s: str) -> Iterator[str]:
    yield from s


def from_split(s: str) -> Iterator[str]:
    yield from s.split()


def flattened(rows: list[list[int]]) -> Iterator[int]:
    for row in rows:
        yield from row


def literal() -> Iterator[int]:
    yield from [1, 2]
    yield from (3, 4)


# --- each operand kind, consumed whole -------------------------------------
print(list(counted(0)), list(counted(3)))
print(list(mixed(2)))
print(list(from_parameter([1, 2])), list(from_parameter([])))
print(list(from_str("ab")))
print(list(from_split("a b  c")))
print(list(flattened([[1, 2], [3], []])))
print(list(literal()))

# --- driven by a for statement, and summed --------------------------------
total = 0
for v in mixed(3):
    total += v
print(total)
print(sum(counted(4)), max(counted(4)), len(list(counted(4))))
