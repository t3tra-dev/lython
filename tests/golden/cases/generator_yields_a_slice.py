# What this pins: a generator that yields a SLICE. `yield xs[i:i + n]` was
# refused with "annotated Iterator[list[int]] but yields builtins.int" --
# inference resolved a slice subscript through `__getitem__`, which for
# `list[int]` answers the ELEMENT, so anything that types an expression without
# emitting it saw a slice as its element type. The generator's yield-type walk
# is exactly that; the same slice in a plain `return` always compiled, because
# that goes through the emitter, which has always used `__getslice__`.
#
# Why this needs to run rather than assert on a diagnostic: the yield type is
# what the suspension lane is built from, so the values have to come back
# whole. A chunker that yielded the first element of each chunk would satisfy
# any diagnostic-only assertion.
#
# Every expected line is python3.14's.

from typing import Iterator


# --- the chunker, which is what a yielded slice is usually for ------------
def chunks(xs: list[int], n: int) -> Iterator[list[int]]:
    i = 0
    while i < len(xs):
        yield xs[i:i + n]
        i += n


for c in chunks([1, 2, 3, 4, 5], 2):
    print(c)
print([len(c) for c in chunks([1, 2, 3, 4, 5, 6], 3)])


# --- constant bounds, an open end, and a step -----------------------------
def slices(xs: list[int]) -> Iterator[list[int]]:
    yield xs[0:2]
    yield xs[2:]
    yield xs[::2]
    yield xs[::-1]


for s in slices([1, 2, 3, 4]):
    print(s)


# --- a str generator, whose slice is a str --------------------------------
def pairs(text: str) -> Iterator[str]:
    i = 0
    while i < len(text):
        yield text[i:i + 2]
        i += 2


print(list(pairs("abcdef")))


# --- the element read must still be the element ---------------------------
# The control: `row[0]` is `__getitem__` and has to stay the element type, or
# "a slice is not an index" would also be satisfied by making nothing one.
# Spelled with a `while` because a generator driven by `for` over a list is a
# separate open gap (tests/probe/known_class_generator_method.py).
def firsts(rows: list[list[int]]) -> Iterator[int]:
    i = 0
    while i < len(rows):
        yield rows[i][0]
        i += 1


print(list(firsts([[1, 2], [3, 4]])))


# --- and a slice outside a generator is unchanged -------------------------
def head(xs: list[int], n: int) -> list[int]:
    return xs[0:n]


print(head([9, 8, 7], 2), head([9, 8, 7], 0))
