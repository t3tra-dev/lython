# What this pins: a class attribute holding a container.
#
#     class R:
#         items: list[str] = []
#     print(R.items)
#     # unsupported static class attribute expression for 'items'
#
# It could not even be READ. The four scalar kinds worked and `cls.n += 1`
# worked, because a scalar class attribute is kept as a compile-time constant
# expression and re-materialized per read. A container cannot be: every read of
# a mutable attribute has to be the SAME object, or a mutation through one is
# invisible to the next.
#
# The mechanism for that already existed -- `classAttrSlots` gives a class
# attribute a module-global CELL, reads and writes going through it -- and
# containers were excluded with the reason "their storage cells would go stale
# against reallocation, the same reason collectModuleGlobals excludes them".
# That reason had stopped being true: collectModuleGlobals gives every contract
# a cell now, because the growth writes THROUGH the handle.
#
# Why this needs to run rather than assert on a diagnostic: sharing is the whole
# point. A per-read copy would print the same first line and diverge only after
# a mutation, so the cases below mutate through the CLASS and read back through
# an INSTANCE, and through a subclass, which is where a copy shows up.
#
# ⛔ Two exclusions, each measured. A `_dunder_` name stays on the constant
# channel: `ctypes.Structure._fields_` is a list the COMPILER consumes, and
# slotting it emits a module-level store that a runtime-internal lib module may
# not have (`stackguard_support.py` stopped building). And a container whose
# ELEMENT type is a union stays there too, for the reason collectModuleGlobals
# already records -- a cell hands back the handle, and a union-typed element read
# needs the literal's per-element evidence.
#
# ⛔ `cls.tags.add(x)` on a set attribute is "set.add requires a rebindable local
# receiver": the mutation rebinds, and the receiver here is an attribute read
# rather than a name. The same set as a module global works, because there the
# receiver IS a name. Reading the attribute works either way, which is what
# this case shows for the set.
#
# Every expected line is python3.14's.


class Registry:
    items: list[str] = []
    counts: dict[str, int] = {}
    tags: set[str] = set()
    pair: tuple[int, int] = (1, 2)
    n: int = 0

    @classmethod
    def add(cls, s: str) -> int:
        cls.items.append(s)
        cls.counts[s] = len(cls.items)
        cls.n += 1
        return cls.n


# --- reading, before anything is mutated ----------------------------------
print(Registry.items, sorted(Registry.counts.items()), sorted(Registry.tags))
print(Registry.pair, Registry.pair[1], Registry.n, len(Registry.items))


# --- mutating through the class, from a classmethod ------------------------
print(Registry.add("a"), Registry.add("b"))
print(Registry.items, sorted(Registry.counts.items()), Registry.n)


# --- the SAME object, read back through an instance -----------------------
r = Registry()
print(r.items, len(r.items), r.n)
Registry.items.append("c")
print(Registry.items, r.items)
r.items.append("d")
print(Registry.items)


# --- inheritance: a base's attribute is shared, a subclass's is its own ----
class Base:
    shared: list[int] = []


class Child(Base):
    own: list[str] = []


Base.shared.append(1)
Child.own.append("x")
print(Base.shared, Child.shared, Child.own)
Child.shared.append(2)
print(Base.shared, Child.shared)


# --- THE CONTROL: the scalar kinds, which always worked -------------------
class Scalars:
    i: int = 0
    s: str = "a"
    f: float = 1.5
    b: bool = True

    @classmethod
    def bump(cls) -> int:
        cls.i += 1
        return cls.i


print(Scalars.i, Scalars.s, Scalars.f, Scalars.b)
print(Scalars.bump(), Scalars.bump(), Scalars.i)
