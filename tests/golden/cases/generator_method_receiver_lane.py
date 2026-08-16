# What this pins: a generator METHOD, called directly on an instance.
#
# It used to be inlined, which substituted the body's own result -- `b.each()`
# typed as None and the for-loop reported that None has no __iter__ -- and was
# then routed through the bound function OBJECT like an async method. That
# reached the refusal this file used to assert: "a generator cannot carry a
# value of contract 'Box' across a suspension yet ... only builtins.int and
# manifest contracts with a rank-1 physical shape have a resume lane, and a
# user class has neither".
#
# ⛔ That reason was wrong, and an EMPTY class refusing said so. The bound
# object captures the receiver in a CLOSURE, and a generator's resume clone
# builds its argument lanes from the callable's POSITIONALS -- so the capture
# had no lane, while the lane for a source class exists and
# `generatorLaneParts` computes it. A direct call takes the SYMBOL now, with
# the receiver as the leading positional, which is the route a recursive method
# already took; as a positional the receiver rides that lane.
#
# Why this needs to run rather than assert on a diagnostic: the receiver has to
# survive every resume, and it is read on each one (`self._n` bounds the loop).
# A frame that dropped it would still compile -- the loop would just stop at a
# different place, or read a freed header.
#
# Every expected line is python3.14's.
from typing import Iterator


class Box:
    def __init__(self, n: int) -> None:
        self._n: int = n

    def each(self) -> Iterator[int]:
        i: int = 0
        while i < self._n:
            yield i
            i = i + 1


b = Box(3)
for v in b.each():
    print(v)


# --- more shapes of the same call: arguments, a str field, a mutated field --
class Bag:
    def __init__(self, xs: list[int]) -> None:
        self.xs = xs

    def each(self):
        for x in self.xs:
            yield x

    def scaled(self, k: int):
        for x in self.xs:
            yield x * k


bag = Bag([1, 2, 3])
print(list(bag.each()))
print(list(bag.scaled(10)))
print(sum(bag.each()) + sum(bag.scaled(2)))


class Counter:
    def __init__(self) -> None:
        self.n = 0

    def ticks(self, k: int):
        i = 0
        while i < k:
            self.n += 1
            yield self.n
            i += 1


counter = Counter()
print(list(counter.ticks(3)), counter.n)
