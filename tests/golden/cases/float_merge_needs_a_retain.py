# What this pins: a merge whose arms are a float and something converted to one.
#
#     print(1.5 if c else 0)
#     # ownership: this block-argument merge needs a retain on the edge and the
#     # header prefix cannot be spelled at the point the retain must go
#
# `1.5 if c else 0.0` was fine, so it was the numeric-tower conversion: the int
# literal becomes a float whose header is an IMMORTAL CONSTANT GLOBAL, and the
# retain writer accepted only a call result or a block argument as
# prefix-initialized. A constant global's initializer IS the prefix -- word 0 is
# INT64_MAX, which is how "immortal" is spelled here -- so the words the retain
# reads are written by construction and the retain it authorises is a no-op on a
# refcount nothing can drive to zero.
#
# The same predicate is what `dict.get` of a float value needed: the absent arm
# carries the union's dead placeholder, which is such a global. That shape is
# here too.
#
# Why this needs to run rather than assert on a diagnostic: what the predicate
# authorises is a RETAIN, and the two ways to get it wrong are a leak and a
# use-after-free -- neither of which changes what a passing program prints. The
# leak gate is this case's other half; the values below only show the merge
# chose the right arm.
#
# ⛔ The merged value's TYPE is `float | int`, which is right -- CPython's false
# arm really is the int 0, and the printed lines below show it. Arithmetic on
# that union is not supported (`bound + 1` is "union<float, int> does not
# provide manifest method '__add__'"), which is the same per-member dispatch `==`
# got and is recorded with it.
#
# ⛔ And a `list[float]` built from those unions is "runtime object header has
# invalid type 'i64'" -- storing a `float | int` where a float element is
# declared is the numeric-tower conversion at a container boundary, not this
# merge. Recorded with the rest.
#
# ⛔ `def f(c: bool) -> float: return 1.5 if c else 0` is still refused
# ("cannot adapt return value to callable return ABI"): returning the merged
# value through a float-declared result is a different adapter. Recorded in
# tests/probe/wb_grid_leftovers_2026_08_16.py.
#
# Every expected line is python3.14's.

flag = True
off = False

# --- the conditional expression, both ways round --------------------------
print(1.5 if flag else 0)
print(1.5 if off else 0)
print(0 if flag else 2.5)
print(2 ** -1 if flag else 0)

bound = 1.5 if flag else 0
print(bound)
other = 0 if off else 3.25
print(other)


# --- a float read out of a dict through .get -------------------------------
scores = {"s": 2.5}
print(scores.get("s"))

mixed = {"s": 2.5, "n": "x"}
print(mixed.get("s"))
print(mixed.get("n"))

record = {"a": 1, "b": "t", "c": 9.5}
print(record.get("c"))


# --- through a function's parameter and a container -----------------------
def pick(flag2: bool) -> float:
    if flag2:
        return 1.5
    return 0.5


print(pick(True), pick(False))


# --- THE CONTROL: two float arms, which always worked ---------------------
print(1.5 if flag else 0.0)
print(2.5 if off else 3.5)
