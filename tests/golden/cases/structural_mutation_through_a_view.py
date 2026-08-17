# What this pins: a structural mutation whose receiver is not a local name.
#
#     self.seen.add(n)         # set.add requires a rebindable local receiver
#     self.rows.insert(i, v)   # list.insert requires a rebindable local
#                              # receiver; bind the list to a local variable
#                              # and insert into the local instead
#     self.rows[:n] = [9]      # slice assignment requires a named local list
#                              # target (field containers are not supported yet)
#     del self.rows[:n]        # slice deletion requires ... (same)
#
# `self.seen.add(x)` is the visited-set idiom, and it did not compile. Nor did
# the same set as a class attribute, a list element, or a dict value; nor insert
# or a slice splice through any of those. Only a plain local name and a
# parameter went through, and `s.add(x)` on a local reached the IDENTICAL
# runtime call.
#
# All four refusals had one premise: a mutation may reallocate, so it hands back
# a re-description that has to be stored somewhere, and only a local can be
# reassigned. That premise expired when list and set became handle-fronted --
# `LySet_AddBox`, `LyList_EnsureCapacity`, `LyList_SetSlice` and
# `LyList_DelSlice` are all void, because the growth writes the new items
# address THROUGH the handle and every holder observes it with nothing to
# rename. A field slot, a class-attribute cell and a container element are
# holders like any other.
#
# Why this needs to run rather than assert on a diagnostic: whether the holder
# observes the mutation is the whole question, and it is a question about a
# LATER read. A version that mutated a copy would compile and print the
# pre-mutation contents, so every section below mutates through one path and
# reads back through another -- through the instance after mutating in a method,
# through the class after mutating via `cls`, through `h[0]` after mutating the
# element.
#
# ⛔ What still holds the rebind: a local name keeps the two-result rebinding
# call, because the rebind is also where the local's element evidence is
# demoted. An interior view gets the one-result call and the evidence lands on
# the receiver value instead. The two shapes are deliberately not collapsed.
#
# Every expected line is python3.14's.


class Graph:
    def __init__(self) -> None:
        self.seen: set[int] = set()
        self.edges: dict[str, set[str]] = {}
        self.order: list[int] = []

    def visit(self, n: int) -> bool:
        if n in self.seen:
            return False
        self.seen.add(n)
        self.order.insert(0, n)
        return True

    def link(self, a: str, b: str) -> None:
        if a not in self.edges:
            self.edges[a] = set()
        self.edges[a].add(b)

    def head(self, n: int) -> int:
        self.order[:n] = [-1]
        return len(self.order)

    def drop(self, n: int) -> int:
        del self.order[:n]
        return len(self.order)


# --- the field receiver: mutate in a method, read through the instance ------
g = Graph()
print(g.visit(1), g.visit(1), g.visit(2))
print(sorted(g.seen), len(g.seen), g.order)
for i in range(4):
    g.visit(i)
print(sorted(g.seen), g.order)

g.link("x", "y")
g.link("x", "z")
g.link("x", "y")
print(sorted(g.edges["x"]), len(g.edges))

print(g.head(2), g.order)
print(g.drop(1), g.order)


# --- the same field, mutated from OUTSIDE the class ------------------------
g.seen.add(99)
g.order.insert(0, 7)
print(sorted(g.seen), g.order)


# --- a field one level down -----------------------------------------------
class Inner:
    def __init__(self) -> None:
        self.tags: set[str] = set()
        self.rows: list[int] = [1, 2, 3]


class Outer:
    def __init__(self) -> None:
        self.inner = Inner()


o = Outer()
o.inner.tags.add("a")
o.inner.tags.add("b")
o.inner.tags.add("a")
o.inner.rows.insert(1, 9)
print(sorted(o.inner.tags), len(o.inner.tags), o.inner.rows)


# --- a class attribute, mutated through `cls` and read through the class ----
class Registry:
    tags: set[str] = set()
    rows: list[int] = [1, 3]

    @classmethod
    def note(cls, s: str) -> int:
        cls.tags.add(s)
        cls.rows.insert(1, len(cls.tags))
        return len(cls.tags)


print(Registry.note("a"), Registry.note("b"), Registry.note("a"))
print(sorted(Registry.tags), Registry.rows)
Registry.rows[:1] = [0]
print(Registry.rows)
del Registry.rows[2:]
print(Registry.rows)


# --- a container element --------------------------------------------------
holders: list[set[str]] = [set(), set()]
holders[0].add("a")
holders[0].add("b")
print(sorted(holders[0]), len(holders[1]))

nested: list[list[int]] = [[1, 2, 3, 4]]
nested[0].insert(1, 9)
print(nested[0])
nested[0][:2] = [0]
print(nested[0])
del nested[0][1:]
print(nested[0])

buckets: dict[str, set[int]] = {"a": set()}
buckets["a"].add(1)
buckets["a"].add(2)
buckets["a"].add(1)
print(sorted(buckets["a"]))


# --- THE CONTROL: the local and the parameter, which always worked ---------
s: set[str] = set()
s.add("a")
s.add("a")
xs = [1, 3]
xs.insert(1, 2)
xs[:1] = [0]
del xs[2:]
print(sorted(s), xs)


def take(t: set[str], ys: list[int]) -> int:
    t.add("z")
    ys.insert(0, 9)
    return len(t) + len(ys)


t0: set[str] = {"y"}
y0 = [1, 2]
print(take(t0, y0), sorted(t0), y0)
