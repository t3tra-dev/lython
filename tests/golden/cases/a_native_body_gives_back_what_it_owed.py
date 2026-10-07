# WHAT: a runtime body that raises half way gives back what it held -- the
# copy `sorted` was sorting, the key and value a dict insertion was handed, the
# element a set insertion was handed, the frozenset being built -- to the frame
# that catches; and an exception caught INSIDE a user `__hash__` or `__lt__`
# those bodies call releases nothing they still hold, so the dicts, sets and
# sorted lists they go on to build are whole.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the leak is only visible
# to the leak gate this case is registered with, and the half that must NOT
# release is only visible as values that survive: a premature release would
# print freed keys or abort on a refcount.

for i in range(50):
    for k in range(4):
        try:
            if k == 0:
                print(sorted([1, "a", i]))
            elif k == 1:
                print({[i]: 2})
            elif k == 2:
                print({[i], [2]})
            else:
                print(frozenset([[i]]))
        except TypeError:
            pass

class K:
    def __init__(self, v: int) -> None:
        self.v = v
    def __hash__(self) -> int:
        try:
            d = {[self.v]: 1}
            print("no", d)
        except TypeError:
            pass
        try:
            raise ValueError("inner")
        except ValueError:
            pass
        return self.v % 7
    def __eq__(self, o: object) -> bool:
        return isinstance(o, K) and o.v == self.v
    def __repr__(self) -> str:
        return f"K({self.v})"
class L:
    def __init__(self, v: int) -> None:
        self.v = v
    def __lt__(self, o: "L") -> bool:
        try:
            sorted([1, "a"])
        except TypeError:
            pass
        return self.v < o.v
    def __repr__(self) -> str:
        return f"L({self.v})"
class Bad:
    def __hash__(self) -> int:
        raise TypeError("bad hash")
for round in range(30):
    d = {K(1): "a", K(2): "b", K(8): "c"}
    s = {K(3), K(10), K(3)}
    t = sorted([L(3), L(1), L(2)])
    try:
        e = {K(5): 1, Bad(): 2}
        print("no", e)
    except TypeError:
        if round == 0:
            print("caught the hash")
    try:
        f = sorted([L(2), L(1)] + [L(0)])
        g = sorted([1, 2, "x"])
        print("no", f, g)
    except TypeError as err:
        if round == 0:
            print(err)
print(sorted(d.values()), sorted(k.v for k in s), t)
print(d[K(8)], K(10) in s, len(s))
