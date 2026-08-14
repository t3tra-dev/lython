# OPEN, and NEW (found 2026-08-15). A set of small ints iterates in INSERTION
# order; CPython iterates it in hash-slot order, which for small ints is the
# value itself, so CPython prints them ascending and Lython prints them as
# written. A stdout divergence with no diagnostic -- the differential's WRONG
# bucket, not a refusal.
#
# MEASURED against python3.14 (./build/bin/lyc):
#
#   print({1, 0}) .............. CPython {0, 1}      lyc {1, 0}     <- this file
#   x: int = 1; y: int = 0
#   print({x, y}) .............. CPython {0, 1}      lyc {1, 0}
#   print({True, False}) ....... CPython {False, True}  lyc {False, True}
#   b: bool = True; c: bool = False
#   print({b, c}) .............. CPython {False, True}  lyc {True, False}
#   print({"b", "a"}) .......... CPython {'b', 'a'}  lyc {'b', 'a'}
#
# The two spellings that AGREE are coincidences and worth not misreading: the
# bool literal pair agrees because insertion order happens to match, and the str
# pair agrees because CPython's string hashes are randomized per process, so
# their slot order is not the sorted order either -- it just happened to be
# insertion order in both.
#
# ⭐ IT IS THE ITERATION, NOT THE STORAGE. Membership and length are right
# (`0 in {1, 0}`, `len`), and the elements are all present; only the order they
# come out in differs. So this is the repr/iteration walk reading the set in
# insertion order rather than walking the table slots.
#
# ⛔ NOT introduced by the bool widening in appendRuntimeSource on 2026-08-15,
# which is what it looked like when it surfaced during that change's
# observability sweep: measured with the pre-change binary and the output is
# byte-identical. Recorded here so the next reader does not re-attribute it.
#
# ⛔ Why NOT "just sort the output": CPython's order is not sorted, it is the
# table's. It coincides with sorted for small ints because a small int hashes to
# itself, and diverges as soon as the values exceed the table size or collide.
# Sorting would agree on this file and disagree on the general case, which is
# the worse failure -- a WRONG that looks fixed.
#
# differential: expect-wrong the point is the stdout divergence

print({1, 0})
print({2, 1, 0})
