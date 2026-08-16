# What this pins: reading a `@property` and using the value WITHOUT binding it
# to a name first.
#
#     r = Path("/x/y")
#     print(str(r.parent))     # SIGSEGV; CPython prints /x
#
# The attribute inference resolves a read against three channels -- instance
# fields, class static attributes, manifest methods -- and a source-class
# property is none of them, so it answered `builtins.object`. That answer is
# what `str(x)` reads to choose its dispatch, and an erased object routes to
# the manifest `object.__str__`, which reads a payload class id a source
# instance's header does not carry. Binding the read to a name first worked,
# because then the SYMBOL carries the type; using it directly did not.
#
# Why this needs to run rather than assert on a diagnostic: the wrong dispatch
# compiled. Nothing in the pipeline objected -- there was no diagnostic to
# assert, only a crash at run time, and the same shape one member type over
# (`p.n` for an int property) was fine the whole time.
#
# Every expected line is python3.14's.


class Leaf:
    def __init__(self, s: str) -> None:
        self._s = s

    def __str__(self) -> str:
        return "leaf:" + self._s


class Node:
    def __init__(self, s: str) -> None:
        self._s = s

    def __str__(self) -> str:
        return self._s

    def __repr__(self) -> str:
        return "Node(" + self._s + ")"

    @property
    def parent(self) -> "Node":
        return Node(self._s + "-p")

    @property
    def leaf(self) -> Leaf:
        return Leaf(self._s)

    @property
    def size(self) -> int:
        return len(self._s)

    def parent_method(self) -> "Node":
        return Node(self._s + "-m")


r = Node("a")

# --- the crash: a property result used directly ---------------------------
print(str(r.parent))
print(str(r.leaf))
print(str(r.parent.parent))
print(repr(r.parent))

# --- the same read bound first, which always worked -----------------------
v = r.parent
print(str(v))
w = r.leaf
print(str(w))

# --- a method returning the same thing, which always worked ---------------
print(str(r.parent_method()))

# --- a property whose type is not a class ---------------------------------
print(r.size, r.parent.size)
print(r.size + 1)

# --- through pathlib, which is where this was found -----------------------
from pathlib import Path

p = Path("/x/y/z.txt")
print(str(p.parent))
print(str(Path("/x/y").parent))
print(p.parent.name, p.name, p.suffix, p.stem)
print(str(Path("a", "b").parent))
