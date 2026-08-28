# WHAT: `a, b = p` and `for a, b in pairs` over a NamedTuple. The unpack walk
# reads its source by index, and a NamedTuple's elements ARE its fields.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: what is checked is which
# field each target got. A read that takes them in the wrong order compiles,
# and for a pair of the same type it produces two plausible numbers.
import sys
from typing import NamedTuple


class Point(NamedTuple):
    x: int
    y: int


class Row(NamedTuple):
    name: str
    count: int


p = Point(1, 2)
a, b = p
sys.stdout.write(str(a) + " " + str(b) + "\n")

# Indexing already worked; it is here as the control the unpack has to agree
# with.
sys.stdout.write(str(p[0]) + " " + str(p[1]) + "\n")

pairs: "list[Point]" = [Point(1, 2), Point(3, 4)]
for x, y in pairs:
    sys.stdout.write(str(x) + "," + str(y) + " ")
sys.stdout.write("\n")

# Fields of different types, where taking them in the wrong order would not
# even type.
rows: "list[Row]" = [Row("a", 1), Row("bb", 2)]
for name, count in rows:
    sys.stdout.write(name + "=" + str(count) + " ")
sys.stdout.write("\n")

r = Row("solo", 9)
label, n = r
sys.stdout.write(label + " " + str(n) + "\n")
