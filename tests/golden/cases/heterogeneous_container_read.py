# What this pins: reading an element out of a container whose element type is
# a UNION. Evidence selection knows exactly which element it picked -- an int
# for `xs[0]` of `[1, "a"]` -- and handed that bundle back where the result
# type is `int | str`, so every consumer of a union read the TAG from lane 0
# and got a member's header memref instead:
#
#     xs = [1, "a"]
#     print(xs[0])
#     # runtime bundle value 0 for 'builtins.bool' has type 'memref<2xi1>',
#     # but ABI expects 'i1'
#
# A record literal -- `{"name": "ann", "age": 30}` -- is the same shape and the
# commonest way to hit it.
#
# Why this needs to run rather than assert on a diagnostic: what the repair
# builds is the union's own lanes, tag first, from the member the evidence
# selected. Getting the TAG wrong picks the other member's lanes, which
# compiles and prints the wrong thing -- an int read as a str header. The
# printed values say which member the tag names, and the isinstance narrowing
# says the tag is readable at all.
#
# ⛔ Each element is read ONCE here, and that is a boundary rather than a
# style. The first read demotes the container's contents evidence (a read
# hands out an alias, so the description cannot travel with it), and a LATER
# union-typed read then falls to the runtime path, which has no
# `builtins.list.__getitem__` that can produce a union -- it would have to
# widen from the stored class id at run time. Recorded in
# tests/probe/wb_grid_leftovers_2026_08_16.py.
#
# Every expected line is python3.14's.

# --- a heterogeneous list -------------------------------------------------
xs = [1, "a"]
first = xs[0]
second = xs[1]
print(first, second)
if isinstance(first, int):
    print("int", first + 1)
if isinstance(second, str):
    print("str", second.upper())


# --- a record literal, which is a heterogeneous dict ----------------------
rec = {"name": "ann", "age": 30}
name = rec["name"]
age = rec["age"]
print(name, age, len(rec))
if isinstance(name, str):
    print(name.upper())
if isinstance(age, int):
    print(age * 2)


# --- three members --------------------------------------------------------
mixed = [1, "b", 2.5]
a = mixed[0]
b = mixed[1]
c = mixed[2]
print(a, b, c)
if isinstance(a, int):
    print("i", a)
if isinstance(b, str):
    print("s", b)
if isinstance(c, float):
    print("f", c)


# --- an ANNOTATED union element type, at module scope and in a function ---
# The annotation is what made this different: an annotated container gets a
# module-global CELL, and a cell hands back the handle without the per-element
# evidence a union read needs. Value binding keeps it.
opt: list[int | None] = [1, None, 3]
print(len(opt), opt[0], opt[1])

table: dict[str, int | None] = {"a": 1, "b": None}
print(len(table), table["a"], table["b"])


def local_union() -> str:
    inner: list[str | None] = ["a", None]
    head = inner[0]
    if isinstance(head, str):
        return head.upper()
    return "-"


print(local_union())


# --- the control: a HOMOGENEOUS read is not a union and is unchanged ------
ints = [10, 20]
print(ints[0] + ints[1])
counts = {"a": 1, "b": 2}
print(counts["a"] + counts["b"])
