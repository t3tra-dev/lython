# What this pins: a module-level name bound ONCE to something that is not a
# literal -- a table, a list, a tuple, an instance -- is readable from a
# function, and every frame sees the same object. Each of these was
# "unresolved name 'T'": only an ANNOTATED module global got storage, and a
# plain `N = 5` worked by re-emitting the literal, which a container cannot do.
#
# Why this needs to run rather than assert on a diagnostic: re-emitting would
# also have compiled. A second `{"a": 1}` is a second dict, so the property
# that separates a cell from a re-emission is that a MUTATION through one name
# is visible through the other -- which only the values below can show. The
# mutations here also cross the boundary in both directions: module scope
# mutates and a function reads, then a function mutates and module scope reads.
#
# ⛔ What is deliberately absent: `D = {}`. An empty literal has no element
# type to infer, so the cell would be `dict[object, object]` and the read is
# refused ("cannot adapt builtins.object return value"). That is the
# pre-existing inference boundary for an empty container, not a property of
# the cell; `D: dict[str, int] = {}` is the spelling that works.
#
# Every expected line is python3.14's.

# --- a table read from a function -----------------------------------------
TABLE = {"a": 1, "b": 2}
ITEMS = [10, 20, 30]
PAIR = (7, "seven")


def look(k: str) -> int:
    return TABLE[k]


def item(i: int) -> int:
    return ITEMS[i]


def pair_tail() -> str:
    return PAIR[1]


print(look("a"), look("b"))
print(item(0), item(2), len(ITEMS))
print(pair_tail(), PAIR[0])


# --- module scope mutates, the function sees it ---------------------------
def total() -> int:
    s = 0
    for x in ITEMS:
        s += x
    return s


print(total())
ITEMS.append(40)
print(total(), len(ITEMS))


# --- a function mutates, module scope sees it -----------------------------
def remember(k: str, v: int) -> None:
    TABLE[k] = v


remember("c", 3)
print(len(TABLE), TABLE["c"], look("c"))


# --- an instance singleton -------------------------------------------------
class Counter:
    def __init__(self, n: int) -> None:
        self.n: int = n

    def bump(self) -> None:
        self.n += 1


COUNTER = Counter(1)


def go() -> int:
    COUNTER.bump()
    return COUNTER.n


print(go(), go(), COUNTER.n)


# --- `global NAME` over a plain literal binding ---------------------------
# A literal bound once is re-emitted at each reference rather than given a
# cell, which is cheaper and correct for a read -- and there is nothing to
# write into, so the counter idiom was "'global COUNT' names a module global
# this compiler does not give storage to". A `global` declaration is the
# strongest statement a program can make that the binding needs storage.
COUNT = 0
FLAG = False
LABEL = "start"


def bump() -> int:
    global COUNT
    COUNT += 1
    return COUNT


def finish(name: str) -> None:
    global FLAG, LABEL
    FLAG = True
    LABEL = name


print(bump(), bump(), COUNT)
print(FLAG, LABEL)
finish("done")
print(FLAG, LABEL)


# --- a name a function does NOT read keeps its value binding --------------
# `local` is bound at module scope and read only here; `shadow` is the same
# spelling as a function's own local, so the function does not capture it and
# the module's binding is untouched.
local = [1, 2]
local.append(3)
print(local)

shadow = "module"


def uses_its_own() -> str:
    shadow = "local"
    return shadow


print(uses_its_own(), shadow)
