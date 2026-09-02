# What: two sibling subclasses in an IMPORTED module, each calling through the
# base the other overrides, and each taking the base as a constructor argument.
# Two separate refusals stood between this program and running: the module's
# classes were emitted one at a time, so whichever subclass came first could
# not see the other's method table; and the argument check asked whether a
# contract name had a DOT in it to decide whether the MRO was worth consulting,
# which is true of every imported class.
#
# WHY THIS IS RUN: which body a dispatch reaches is a runtime fact, and a
# compiler resolving the base's body would print the same shape with `?` in
# place of the subclass's letter. The decode is that the letters nest -- `LR?`
# is three dispatches deep, one per class.
import a_module_of_sibling_subclasses as tree

t: "tree.Base" = tree.Left(tree.Right(tree.Base()))
print(t.show())

xs: "list[tree.Base]" = [
    tree.Base(),
    tree.Left(tree.Base()),
    tree.Right(tree.Left(tree.Base())),
]
print([x.show() for x in xs])
