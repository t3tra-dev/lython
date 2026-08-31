# The tree walk every iterable tree is written as -- a generator method that
# recurses into its children -- is refused, and until 2026-09-01 it was refused
# for the WRONG REASON:
#
#     static type !py.contract<"T"> does not provide manifest method 'walk'
#
# pointing at the `def`, for a method the class plainly declares. Computing a
# generator's signature WALKS its body to infer the yield type, and the
# recursive `k.walk()` in there resolved against a table that did not have
# `walk` yet -- the publication happens after. The same method called from a
# SIBLING method compiled, because a sibling body runs after the whole class
# walk.
#
# ⭐ NOW IT SAYS THE REAL LIMIT: "yield from delegation exceeded the static
# inlining budget (recursive delegation has no static expansion)". A generator
# that delegates to itself has no bounded static expansion, which is a property
# of the delegation strategy and not of this class.
#
# The repair for the message: a generator method that names itself is published
# from the FIRST signature pass and its signature recomputed. With a return
# annotation the published entry is exact (the annotation, not the inference --
# a generator's public result IS the inferred generator type, which is the very
# thing the missing entry spoiled); without one it publishes the first pass's
# `object` and the second pass ends there, which is the boundary the
# progressive publication already documents for two unannotated methods that
# call each other.
#
# ⛔ WHAT IS STILL MISSING is runtime delegation: `yield from` is expanded
# statically, so recursion needs a `yield from` that resumes a delegate at run
# time rather than inlining it. That is the mechanism, and it is the same one
# `wb_generator_iterates_a_dict.py` needs for a different reason.
from typing import Iterator


class Tree:
    def __init__(self, value: int) -> None:
        self.value = value
        self.children: list["Tree"] = []

    def walk(self) -> "Iterator[int]":
        yield self.value
        for child in self.children:
            for nested in child.walk():
                yield nested


root = Tree(1)
root.children.append(Tree(2))
print(list(root.walk()))
