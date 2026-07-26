# Grow a container that lives one level DOWN a call-derived instance:
# `self.leaf.xs.append(v)`, where `leaf` is a field holding an instance whose own
# field holds the list. Both reads are interior views, and the mutation has to
# reach the inner slot.
#
# Two things are asserted, and the first is why this file exists separately from
# the single-level cases. The old lowering had no arm for a non-rebindable
# receiver without element evidence, so it used the evidence tier, whose
# insertion index is "how many elements do I know about" -- zero for a list read
# out of a slot. It appended over element zero and set the length to one, so the
# list printed `[3, 2, <object object at ...>]`: a wrong value, a wrong length,
# and a slot read past the end.
#
# Second: the same program with the receiver built in this frame instead of
# returned from a factory. It used to be the discriminator for this shape, so
# both spellings are here and must agree.


class Leaf:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs


class Tree:
    def __init__(self, leaf: Leaf) -> None:
        self.leaf: Leaf = leaf

    def grow(self, v: int) -> None:
        self.leaf.xs.append(v)


def mk() -> Tree:
    return Tree(Leaf([1, 2]))


t = mk()
t.grow(3)
t.grow(4)
print(t.leaf.xs)

inner: Leaf = Leaf([1, 2])
u = Tree(inner)
u.grow(3)
print(u.leaf.xs)

# Binding the intermediate to a local first is the same chain with the middle
# view named, which used to fail the same way.
v = mk()
mid: Leaf = v.leaf
mid.xs.append(9)
print(v.leaf.xs)
print(len(mid.xs))
