# What this pins: reading a local after it was stored into an object field.
#
#     seed = [3, 1, 2]
#     b = Bag(seed)          # __init__ does self.xs = xs
#     i = 0
#     while i < 3:
#         print(len(seed))   # printed 0 0 0; CPython prints 3 3 3
#         i += 1
#
#     for v in seed: ...     # SIGSEGV
#     [v for v in seed]      # []
#     max(seed)              # ValueError: max() iterable argument is empty
#
# A use-after-free. The field store retains for the slot and then releases the
# value's OWN token, which is right for a temporary -- `self.xs = [1, 2, 3]` has
# the two cancel and the slot inherit the single reference -- and wrong for a
# local the caller keeps reading. The list was freed at the store; every later
# read through the name found whatever the next allocation had written there,
# and a length of 0 is what an empty-then-reused handle reads as.
#
# It hid well. Nothing allocates between the release and a read at module scope,
# so `print(len(seed))` right after the constructor answered 3; reading the field
# in the same statement (`print(len(seed), len(b.xs))`) answered 3 3; and a later
# `print(b.xs)` moved the release past the loop and made the whole program right.
# Only a read separated from the store by an allocation -- which is what a loop,
# a comprehension or a `print` of anything else is -- showed it.
#
# The test is now "does this store dominate a use of the value", asked on the
# PY-level operand: the walk lowers in program order, so a later read is still
# an unlowered `py.len`, and the physical handle has no uses past the store yet.
#
# Why this needs to run rather than assert on a diagnostic: nothing was
# diagnosed. Two of the four shapes printed a number, one printed `[]` and one
# faulted, all from the same freed handle -- so what is pinned is the values a
# read produces at a distance from the store, with an allocation in between.
#
# Every expected line is python3.14's.


class Bag:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs


class Pair:
    def __init__(self, d: dict[str, int], t: set[int]) -> None:
        self.d: dict[str, int] = d
        self.t: set[int] = t


# --- the length in a loop condition and in a loop body ---------------------
seed = [3, 1, 2]
b = Bag(seed)
i = 0
while i < 3:
    print(len(seed))
    i += 1

j = 0
while j < len(seed):
    print(seed[j])
    j += 1


# --- iteration, a comprehension, and the aggregate builtins ---------------
s2 = [3, 1, 2]
b2 = Bag(s2)
for v in s2:
    print(v)
print([v for v in s2])
print(sum(s2), max(s2), min(s2), sorted(s2))


# --- a str element, so a freed element shows as well as a freed handle -----
s3 = ["a", "bb"]


class Words:
    def __init__(self, xs: list[str]) -> None:
        self.xs: list[str] = xs


w = Words(s3)
for x in s3:
    print(x, len(x))


# --- inside a function, where the local dies at the end -------------------
def run() -> int:
    local = [3, 1, 2]
    holder = Bag(local)
    total = 0
    for v in local:
        total += v
    return total + len(holder.xs)


print(run())


# --- a dict and a set through the same store ------------------------------
d0 = {"a": 1, "b": 2}
t0 = {7, 8}
p = Pair(d0, t0)
for k in sorted(d0):
    print(k, d0[k])
print(len(d0), sorted(t0), 7 in t0)


# --- the store's own reason still holds: a temporary is not kept alive -----
# `Bag([4, 5])` has no other name, so the slot inherits the single reference and
# nothing leaks. The leak gate is the other half of this line.
temp = Bag([4, 5])
print(temp.xs, len(temp.xs))
for v in temp.xs:
    print(v)
