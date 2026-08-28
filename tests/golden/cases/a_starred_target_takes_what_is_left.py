# WHAT: `first, *rest = xs` and its two other shapes -- the star at the end, at
# the front, and in the middle -- over a list, a tuple, a string and a
# heterogeneous tuple, plus the ValueError a source too short for the fixed
# targets raises.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the star's share is a
# SLICE with a bound counted from each end, and every way of getting that wrong
# produces a value rather than a diagnostic -- an off-by-one at either end
# silently moves an element between `rest` and the target beside it, and the
# star's value has to be a LIST even when the source was a tuple or a string,
# which only shows in what is printed.
import sys


def line(text: str) -> None:
    sys.stdout.write(text + "\n")


first, *rest = [1, 2, 3, 4]
line(str(first) + " " + str(rest))

*init, last = [1, 2, 3, 4]
line(str(init) + " " + str(last))

a, *mid, z = [1, 2, 3, 4, 5]
line(str(a) + " " + str(mid) + " " + str(z))

# A tuple source still leaves a list.
p, *q = (7, 8, 9)
line(str(p) + " " + str(q))

# The star takes nothing when the source is exactly the fixed targets.
only, *empty = [5]
line(str(only) + " " + str(empty))

# A string is a sequence of one-character strings.
c, *cs = "abc"
line(c + " " + str(cs))

# ⛔ A HETEROGENEOUS TUPLE IS SPELLED OUT rather than sliced: its arity is in
# its type, and slicing it would type the elements as their union.
pairs: "list[tuple[int, str]]" = [(1, "a"), (2, "b")]
for n, *names in pairs:
    line(str(n) + " " + str(names))

# ⛔ THE ORIGINAL SILENT DEFECT, kept: the star was skipped by the indexed
# walk, so a target that already held something kept it. It only showed up
# where the name existed beforehand -- anywhere else the program was refused
# for the unrelated "unresolved name".
xs: "list[int]" = [1, 2, 3, 4]
kept: "list[int]" = [7, 7, 7]
head, *kept = xs
line(str(head) + " " + str(kept))

# Not enough values for the fixed targets is CPython's ValueError, with the
# count it expected at least and the count it got.
try:
    x, *y, w = [1]
except ValueError as e:
    line("V: " + str(e))

try:
    empty_head, *empty_tail = []
except ValueError as e:
    line("V: " + str(e))
