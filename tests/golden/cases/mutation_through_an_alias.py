# What this pins: reading a list by one name after mutating it through another.
#
#     seed = [3, 1, 2]
#     b = Bag(seed)        # self.xs = xs -- one object, two names
#     b.xs[0] = 9
#     print(seed[0])       # printed 3; CPython prints 9
#     print(seed)          # [9, 1, 2] -- right all along
#
# The write always landed. Only the ELEMENT READ through the other name was
# wrong, and it was wrong because it never touched the runtime: a list built from
# a literal carries compile-time slot evidence, and a mutation through a second
# holder has no way to reach the first name's copy of it. `b.xs.sort()` printed
# the pre-sort element, `holder[0][0] = 9` the pre-write one, and
# `b.xs.append(9)` then `seed[3]` did not compile at all ("owned resource ... is
# released or transferred more than once"), because two names each thought they
# owned the slot.
#
# The mark for this already existed and was already set at every absorption
# (`sharedWithHolder`, set when a container goes into a literal, a slot or a
# field). Nothing consulted it on the READ side. It does now: a list with a
# second holder answers `[i]` from the payload.
#
# Why this needs to run rather than assert on a diagnostic: three of the four
# shapes printed a plausible number. The list itself printed correctly in every
# one of them, which is why `repr` is here only as a witness that the mutation
# landed -- what is pinned is the single-element read next to it.
#
# ⛔ The evidence is kept, not dropped. Dropping it at the absorption was
# measured three ways and takes 145-146 tests down, aborting at runtime: the
# evidence is where a slot's owned reference is BOOKED, not only what it
# describes. So the read moves and the bookkeeping stays.
#
# This file replaces errors/list_insert_on_field, whose own note said what would
# retire it -- "when stage 4b puts the payload behind the handle, growth stops
# re-rooting anything and this refusal should become unnecessary; this file is
# how that gets noticed". Its program is the first section below.
#
# Every expected line is python3.14's.


class Bag:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs


# --- the old errors/ program: insert through the field, read through the local
seed = [3, 1, 2]
b = Bag(seed)
b.xs.insert(0, 9)
print(b.xs, seed)
print(len(b.xs), len(seed), b.xs[0], seed[0], seed[1])

# and the other direction
seed.insert(1, 8)
print(b.xs, seed, b.xs[1], seed[1])


# --- __setitem__ through the field ----------------------------------------
s1 = [3, 1, 2]
b1 = Bag(s1)
b1.xs[0] = 9
print(s1[0], b1.xs[0], s1)


# --- append, which did not compile ----------------------------------------
s2 = [3, 1, 2]
b2 = Bag(s2)
b2.xs.append(9)
print(len(s2), s2[3], s2, b2.xs)


# --- an in-place permutation ----------------------------------------------
s3 = [3, 1, 2]
b3 = Bag(s3)
b3.xs.sort()
print(s3[0], s3[2], s3, b3.xs)

s4 = [1, 2, 3]
b4 = Bag(s4)
b4.xs.reverse()
print(s4[0], s4, b4.xs)


# --- pop, which was always right (it does not move a surviving slot) -------
s5 = [3, 1, 2]
b5 = Bag(s5)
print(b5.xs.pop(), len(s5), s5[0], s5)


# --- a holder that is a container, not an object ---------------------------
s6 = [3, 1, 2]
holder = [s6]
holder[0][0] = 9
print(s6[0], s6)

s7 = [3, 1, 2]
holder2 = [s7]
holder2[0].insert(0, 7)
print(s7[0], s7[1], len(s7), s7)


# --- a dict value ---------------------------------------------------------
s8 = [3, 1, 2]
by_name = {"a": s8}
by_name["a"][1] = 9
print(s8[1], s8)


# --- THE CONTROL: an unaliased list still reads from its evidence ----------
# Nothing else holds `plain`, so the compile-time slots are authoritative and
# these reads must keep answering.
plain = [3, 1, 2]
plain[0] = 9
print(plain[0], plain[1], plain[2], plain)
