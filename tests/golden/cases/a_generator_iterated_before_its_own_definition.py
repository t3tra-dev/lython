# WHAT: a generator that holds one of two generators in a variable and
# delegates to it, where the second is defined BELOW it and iterates a
# generator method of a class defined below that.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the second generator's
# yields could not be typed when the first was read (what it iterates is not
# published yet), so it read as yielding None and the two joined to nothing
# a `yield from` takes; it is annotated `-> Iterator[str]`, which is what it
# now reads as. Which generator ran is seen only in what comes out.
from collections.abc import Iterator


def first() -> Iterator[str]:
    yield "a"


def pick(flag: bool) -> Iterator[str]:
    if flag:
        g = first()
    else:
        g = later(3)
    yield from g


def later(n: int) -> Iterator[str]:
    for group in Source(n).groups():
        yield "group of %d" % len(group)


class Source:
    def __init__(self, n: int) -> None:
        self.n = n

    def groups(self) -> Iterator[list[int]]:
        for i in range(1, self.n + 1):
            yield list(range(i))


print(list(pick(True)), list(pick(False)))
