# probe: grow a list that lives ONE LEVEL DOWN a call-derived instance
# axes: acquire=call width=w3list op=nested-append flow=straight
# CLASSIFICATION: 4 クラッシュ / abort
#   'repr: boxed element has no conforming __repr__' のあと SIGABRT
# CPython 3.14 expects: [1, 2] then [1, 2, 3]
#
# PRE-EXISTING, not a stage-4a regression: identical failure on main at ed6a798.
# No probe reached it because the rebind family stores INTO a field and the
# alias-read family reads a field of the receiver itself; this one grows a list
# owned by an instance the receiver's field holds. Found by the integration
# track's nested-chain golden.
#
# The control is nested_field_grow_inline.py: the same program with the receiver
# built in this frame passes, so the discriminator is the acquisition path (the
# owned-local marker gap), not the chain depth -- a two-level chain is enough,
# and binding the intermediate to a local first (`inner = t.leaf;
# inner.xs.append(3)`) fails the same way.

class Leaf:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs


class L1:
    def __init__(self, leaf: Leaf) -> None:
        self.leaf: Leaf = leaf

    def grow(self, v: int) -> None:
        self.leaf.xs.append(v)


def mk() -> L1:
    return L1(Leaf([1, 2]))


t = mk()
print(t.leaf.xs)
t.grow(3)
print(t.leaf.xs)
