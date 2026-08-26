# WHAT: a class whose field is a union with itself -- `nxt: Optional["Node"]`,
# the shape every linked structure is written in -- links, reads back, and ends
# at None.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the field is stored as a
# BOX, so the arm is not in the value -- it is the entity word being zero or
# not. Every assertion here is about which arm a read reconstructed, and only
# the running program says. `print(n1 is None)` decodes the empty box, `n1.v`
# decodes the full one, and the last line reads a box that was never written.
from typing import Optional


class Node:
    v: int
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.nxt = None


a = Node(1)
b = Node(2)
c = Node(3)
a.nxt = b
b.nxt = c

n1 = a.nxt
print(n1 is None)
if n1 is not None:
    print(n1.v)
    n2 = n1.nxt
    if n2 is not None:
        print(n2.v, n2.nxt is None)

a.nxt = None
print(a.v, a.nxt is None)
