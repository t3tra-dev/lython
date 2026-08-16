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
# Reading the SAME container in more than one block is the second half. The
# contents evidence used to be dropped at every op outside the block that
# defines the container's storage, which is right for a container something can
# mutate and wrong for one nothing can: printing a union branches on the tag, so
# `print(xs[0]); print(xs[1])` puts the second read in a successor block, and
# the runtime tier it fell to has no `builtins.list.__getitem__` that can
# produce a union. A container every use of which is a read describes the same
# contents everywhere.
#
# And an element is stored in the SLOT's form, which for `bool` is a boxed
# header rather than the canonical i1 its ABI names. The union lane is the i1
# and the injection counts values rather than checking them, so the header went
# into the lane -- bool is the only contract with a `box` primitive, which is
# why every other element type worked.
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


# --- the same container read from more than one block ---------------------
# Every print between these reads branches on a tag, so each later read is in a
# block the container's definition merely dominates.
row = [1, "ann", 2.5, True]
print(row[0])
print(row[1])
print(row[2])
print(row[3])
print(row[-1])
print(len(row))

card = {"name": "ann", "age": 30, "score": 9.5, "member": True}
print(card["name"])
print(card["age"])
print(card["score"])
print(card["member"])
print(len(card))

lv = card["age"]
if isinstance(lv, int):
    print(lv + 1)


# --- a bool member, which is the boxed one --------------------------------
flags = [True, "on"]
print(flags[0])
print(flags[1])

single = {"ok": True, "why": "fine"}
print(single["ok"])
print(single["why"])


# --- inside a function, called twice --------------------------------------
def show(tag: str) -> None:
    rec = {"k": 1, "v": "x", "f": False}
    print(tag, rec["k"])
    print(tag, rec["v"])
    print(tag, rec["f"])


show("a")
show("b")


# --- THE CONTROL: a container something MUTATES still reads through the
# runtime, so the evidence may not survive the block it was built in --------
# ⛔ The mutated one here is HOMOGENEOUS, and that is the remaining boundary
# rather than a simplification: a heterogeneous container mutated across a
# block boundary has nowhere to go. Its evidence is gone by the rule above and
# the runtime tier still has no `__getitem__` that can produce a union --
# closing that means widening from the stored class id at run time. Recorded in
# tests/probe/wb_grid_leftovers_2026_08_16.py.
grow = [1, "a"]
grow.append(2)
print(grow[0])
print(len(grow))

counts2 = {"a": 1, "b": 2}
print(counts2["a"])
counts2["c"] = 3
print(len(counts2), counts2["c"])
