# WHAT: str.join over a generator handed back by a function annotated
# `-> Iterator[str]` -- difflib's `"".join(ndiff(a, b))`, where ndiff returns
# Differ.compare's generator -- beside join over a generator called in place.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the returned generator
# reached the lowering as itself ("cannot adapt types.GeneratorType to
# runtime input 2 of builtins.str.join") because only a generator CALLED in
# the argument was read into a list first. Whether every value arrives, in
# order, is seen in the joined text.
from collections.abc import Iterator


class Lines:
    def __init__(self, words: list[str]) -> None:
        self.words = words

    def each(self) -> Iterator[str]:
        for w in self.words:
            yield w.upper()

    def handed_back(self) -> Iterator[str]:
        return self.each()


def through_a_function(words: list[str]) -> Iterator[str]:
    return Lines(words).each()


print("-".join(Lines(["a", "b", "c"]).handed_back()))
print("".join(through_a_function(["x", "y"])), ",".join(Lines(["p"]).each()))
