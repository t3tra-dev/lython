# WHAT: a loop whose carried Optional starts as a BORROWED parameter and whose
# body writes an Optional value into an Optional FIELD -- relinking a chain in
# place, and the same function with nothing to relink.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the loop is lent a
# reference on its entry edge because a parameter carries none of its own, and
# that lend is discharged where the value dies. Both ends were wrong and
# neither says so where it goes wrong: the lend outlived the function when the
# body never ran, and the field write was wrapped in a region the release
# planner will not place into. What decides it is the structure the CALLER
# still holds afterwards, so every line prints the chain from the caller's side
# once the callee has returned -- and each function is called twice on the same
# nodes, because one missing reference only shows on the second visit.
from typing import Optional


class Node:
    v: int
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.nxt = None


def length(head: Optional[Node]) -> int:
    n = 0
    cur = head
    while cur is not None:
        n += 1
        cur = cur.nxt
    return n


def relink_first(head: Node, tail: Optional[Node]) -> int:
    seen = 0
    cur: Optional[Node] = head
    while cur is not None:
        seen += 1
        if cur.v == 0:
            cur.nxt = tail
        cur = cur.nxt
    return seen


a = Node(0)
b = Node(1)
c = Node(2)
b.nxt = c

print(relink_first(a, b), length(a), a.v, b.v, c.v)
print(relink_first(a, b), length(a), a.v)
print(relink_first(a, c), length(a), a.v, c.nxt is None)
print(relink_first(a, None), length(a), a.v, a.nxt is None)
print(relink_first(a, None), length(a), b.v, c.v)
