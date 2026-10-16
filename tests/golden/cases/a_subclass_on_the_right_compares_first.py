# Why execution: the answer is WHICH BODY ran, and both bodies return a bool
# the compiler accepts; only the printed values tell `a.__eq__(b)` from
# `b.__eq__(a)`.
#
# A right operand whose class strictly derives the left's is asked first, as
# CPython's do_richcompare does: at an operator written in the program (a
# synthesized two-operand dispatch, since a Base-typed value may hold a Sub),
# and inside a container's comparison (tuple and list equality, dict and set
# lookup, list.index), which the runtime answers. A subclass that only
# INHERITS the compared method is asked first too, with itself as `self`.
class Base:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Base) and self.tag.startswith("B")

    def __lt__(self, other: "Base") -> bool:
        return self.tag < other.tag

    def __hash__(self) -> int:
        return 1


class Sub(Base):
    def __eq__(self, other: object) -> bool:
        return False

    def __gt__(self, other: "Base") -> bool:
        return True

    def __hash__(self) -> int:
        return 2


class Leaf(Sub):
    pass


class Side(Base):
    pass


class Plain(Base):
    def __ne__(self, other: object) -> bool:
        return True


def show(x: Base, y: Base) -> None:
    print(x.tag, y.tag, x == y, x != y, x < y, x > y)


items: list[Base] = [Base("Bx"), Sub("S"), Leaf("L"), Side("Bs"), Plain("Bp")]
for x in items:
    for y in items:
        show(x, y)


class K:
    def __init__(self, n: int) -> None:
        self.n = n

    def __eq__(self, other: object) -> bool:
        return isinstance(other, K) and other.n == self.n

    def __hash__(self) -> int:
        return 7

    def __lt__(self, other: "K") -> bool:
        return self.n < other.n


class Strict(K):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Strict) and other.n == self.n

    def __hash__(self) -> int:
        return 7

    def __gt__(self, other: "K") -> bool:
        return False


k: K = K(1)
st: K = Strict(1)
print((k,) == (st,), (st,) == (k,), [k] == [st], [st] == [k])
d: dict[K, str] = {k: "k"}
print(st in d, k in d, d.get(st, "none"))
d2: dict[K, str] = {st: "s"}
print(k in d2, st in d2)
s1: set[K] = {k}
print(st in s1)
xs: list[K] = [K(3), Strict(2), K(1)]
print([v.n for v in sorted(xs)])
print([K(1), Strict(1)].index(Strict(1)))
print(k != st, st != k)
