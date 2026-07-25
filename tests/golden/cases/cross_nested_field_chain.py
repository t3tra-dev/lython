# Four levels of user-class-typed fields, read and written through the chain.
# The 4a implementer reported two defects with one cause: a read decided whether
# to retain from the field's CONTRACT while only one- and two-level chains were
# tested, so `t.m.i.n` read back 0; and a nested field read was classified as a
# method call because `attr.get`'s possible shapes were never enumerated. This
# pins the shapes those two would have caught, at a depth neither branch's own
# case reaches.
#
# Every form here uses a CALL-DERIVED receiver (`mk()`), which is the half of
# the acquisition axis that has no owned-local marker. The forms are: a chained
# read at four levels; a chained read inside a method; a store at the bottom of
# the chain from a method and from a borrowed parameter; a rebind of the
# next-to-last link and of the first link, both from a callee; two aliases of
# intermediate links agreeing with the chain after a store through one of them;
# and the whole thing as a container element.
#
# NOT here: growing a list that lives one level down (`self.leaf.xs.append(v)`),
# which aborts on a call-derived receiver. That shape predates this wave -- it
# fails identically on main at ed6a798 -- and is pinned in the corpus as
# probe/nested_field_grow_call.py against its passing inline control.


class Leaf:
    def __init__(self, n: int, s: str, xs: list[int]) -> None:
        self.n: int = n
        self.s: str = s
        self.xs: list[int] = xs


class L3:
    def __init__(self, leaf: Leaf) -> None:
        self.leaf: Leaf = leaf


class L2:
    def __init__(self, c: L3) -> None:
        self.c: L3 = c


class L1:
    def __init__(self, b: L2) -> None:
        self.b: L2 = b

    def depth(self) -> int:
        return self.b.c.leaf.n

    def word(self) -> str:
        return self.b.c.leaf.s

    def sink(self, v: int) -> None:
        self.b.c.leaf.n = v


def mk() -> L1:
    return L1(L2(L3(Leaf(5, "ab", [1, 2]))))


t = mk()
print(t.b.c.leaf.n, t.b.c.leaf.s, t.b.c.leaf.xs)
print(t.depth(), t.word())
t.sink(9)
print(t.b.c.leaf.n, t.depth())


def poke(x: L1) -> None:
    x.b.c.leaf.n = 42
    x.b.c.leaf.s = "cd"


poke(t)
print(t.b.c.leaf.n, t.b.c.leaf.s, t.depth(), t.word())


def replace_leaf(x: L1) -> None:
    x.b.c.leaf = Leaf(0, "zz", [7])


replace_leaf(t)
print(t.b.c.leaf.n, t.b.c.leaf.s, t.b.c.leaf.xs, t.depth())


def replace_mid(x: L1) -> None:
    x.b = L2(L3(Leaf(1, "q", [8, 9])))


replace_mid(t)
print(t.b.c.leaf.n, t.b.c.leaf.s, t.b.c.leaf.xs)

mid = t.b
inner = mid.c
print(inner.leaf.n, inner.leaf.s)
inner.leaf.n = 11
print(t.b.c.leaf.n, mid.c.leaf.n, inner.leaf.n)

nested: list[L1] = [mk(), mk()]
nested[0].sink(21)
print(nested[0].b.c.leaf.n, nested[1].b.c.leaf.n)
print(nested[0].depth(), nested[1].word())
