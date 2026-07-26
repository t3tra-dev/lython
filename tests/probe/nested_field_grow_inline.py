# probe: control for nested_field_grow_call.py -- same program, receiver built
# in this frame instead of returned by a factory
# axes: acquire=inline width=w3list op=nested-append flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: [1, 2] / [1, 2, 3]
#
# The reason this file exists: without it the abort in nested_field_grow_call.py
# could be read as "growing a list one level down is unimplemented". It is not
# -- the same source compiles and runs correctly here, so what the abort is
# about is the receiver's acquisition path.

class Leaf:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs


class L1:
    def __init__(self, leaf: Leaf) -> None:
        self.leaf: Leaf = leaf

    def grow(self, v: int) -> None:
        self.leaf.xs.append(v)


t = L1(Leaf([1, 2]))
print(t.leaf.xs)
t.grow(3)
print(t.leaf.xs)
