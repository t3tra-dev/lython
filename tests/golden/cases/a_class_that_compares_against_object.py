# WHAT: `__eq__(self, other: object)` and `__lt__(self, other: object)` -- the
# spelling Python's own protocol is written in -- reached through the paths
# that compare ERASED values: `in` over a list, `index`/`count`, a dict lookup,
# set de-duplication, and `sorted`.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the dispatch hands the
# method its two operands out of two boxes, and what the second one IS was
# decided by comparing the two parameters' physical SHAPES. A class with three
# body words is five i64 words, and so is a box, so this signature looked
# symmetric and the method was handed the box's entity where a box belonged.
# It read a field handle as an entity pointer: an intermittent SIGSEGV, and on
# the runs that did not crash, `False`.
import sys


class Key:
    n: int

    def __init__(self, n: int) -> None:
        self.n = n

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Key) and other.n == self.n

    def __ne__(self, other: object) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash(self.n)

    def __repr__(self) -> str:
        return "Key(" + str(self.n) + ")"


class Ranked:
    n: int

    def __init__(self, n: int) -> None:
        self.n = n

    def __lt__(self, other: object) -> bool:
        return isinstance(other, Ranked) and self.n < other.n

    def __repr__(self) -> str:
        return "R" + str(self.n)


keys: "list[Key]" = [Key(1), Key(2), Key(1)]
sys.stdout.write(str(Key(1) in keys) + " " + str(Key(9) in keys) + "\n")
sys.stdout.write(str(keys.index(Key(2))) + " " + str(keys.count(Key(1))) + "\n")
sys.stdout.write(str(Key(1) == Key(1)) + " " + str(Key(1) != Key(2)) + "\n")
sys.stdout.write(repr(keys) + "\n")

seen: "set[Key]" = {Key(1), Key(2), Key(1)}
sys.stdout.write(str(len(seen)) + "\n")

table: "dict[Key, str]" = {}
table[Key(1)] = "one"
table[Key(2)] = "two"
# A key EQUAL to the stored one but not the same object: the probe has to
# compare, which is the path that crashed.
sys.stdout.write(table[Key(1)] + " " + str(Key(2) in table) + " "
                 + str(Key(3) in table) + "\n")

ranked: "list[Ranked]" = [Ranked(3), Ranked(1), Ranked(2)]
sys.stdout.write(repr(sorted(ranked)) + "\n")

# The same question against something that is NOT this class, which is what the
# `object` annotation is for.
sys.stdout.write(str(Key(1) == "not a key") + " " + str(Key(1) != 7) + "\n")
