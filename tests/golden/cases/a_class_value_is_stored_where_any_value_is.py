# WHAT: a class value kept in each kind of storage -- a module global a
# function rebinds through `global`, a cell a nested function rebinds through
# `nonlocal`, a field of a union type, a generator's yielded values, a tuple,
# a dict keyed by class and built up by item assignment -- reads back as the
# class that was stored last.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: each store writes the
# class value's type-object address into its own storage and each read
# rebuilds the value from there; only printing what came back shows that the
# write reached the read. Several of these used to print the class stored
# FIRST: the global was an alias every read folded to A, the `nonlocal` cell
# wrote a word where its reader looked for a box, and the union field took a
# store it had not wrapped.
class A:
    pass


class B(A):
    pass


current: type[A] = A


def switch() -> None:
    global current
    current = B


print(current.__name__)
switch()
print(current.__name__)


def outer() -> str:
    k: type[A] = A

    def flip() -> None:
        nonlocal k
        k = B

    flip()
    return k.__name__


print(outer())


class Holder:
    def __init__(self) -> None:
        self.kind: type[A] | None = None


h = Holder()
print(h.kind is None)
h.kind = B
k = h.kind
if k is not None:
    print(k.__name__)


def walk(n: int):
    for i in range(n):
        if i % 2 == 0:
            yield A
        else:
            yield B


print([c.__name__ for c in walk(3)])


def under(cls: type[A]):
    for c in [A, B]:
        if issubclass(c, cls):
            yield c


print([c.__name__ for c in under(B)], [c.__name__ for c in under(A)])
pair = (A, B)
a, b = pair
print(a.__name__, b.__name__, pair[1] is B, len(pair))
counts: dict[type, int] = {}
for v in [1, "s", 2, "t", 3.0]:
    t = type(v)
    counts[t] = counts.get(t, 0) + 1
print(sorted((c.__name__, n) for c, n in counts.items()))
