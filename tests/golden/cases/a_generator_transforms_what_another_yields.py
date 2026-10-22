# WHAT: a generator iterating another generator and yielding something made
# from each value -- a str, a sum, a list, through two levels, stopped early,
# and raising part way -- consumed whole and stepped with next().
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: each of these failed in
# the lowering ("operand #0 does not dominate this use"): the inner
# generator's step read the generator its call made, not the one the outer
# frame carried across a yield. Only yielding the value unchanged worked,
# because that is rewritten as `yield from`. Whether the resumed inner
# generator is the right one is seen in the sequence of values.
from collections.abc import Iterator


def inner(n: int) -> Iterator[int]:
    for i in range(n):
        yield i * 10


def as_text(n: int) -> Iterator[str]:
    for v in inner(n):
        yield "v%d" % v


def plus_one(n: int) -> Iterator[int]:
    for v in inner(n):
        yield v + 1


def wrapped(n: int) -> Iterator[list[int]]:
    for v in inner(n):
        yield [v]


def two_levels(n: int) -> Iterator[str]:
    for s in as_text(n):
        for t in inner(2):
            yield s + ":" + str(t)


def stops(n: int) -> Iterator[int]:
    for v in inner(n):
        if v > 20:
            break
        yield v + 1
    yield -1


def fails(n: int) -> Iterator[int]:
    for v in inner(n):
        if v == 20:
            raise ValueError("at %d" % v)
        yield v


print(list(as_text(3)), list(plus_one(3)), list(wrapped(2)))
print(list(two_levels(2)))
print(list(stops(6)))
try:
    for x in fails(5):
        print("got", x)
except ValueError as e:
    print("caught", e)
g = two_levels(3)
print(next(g), next(g), next(g))
