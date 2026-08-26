# WHAT: `break` out of a loop that carries an Optional -- the shape a bounded
# walk over a linked structure is written in (`traceback.print_tb`'s `limit`
# is this exact program).
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defect was an
# over-release, and an over-release has no diagnostic -- it is a refcount that
# reaches zero while a name still holds the object. What makes it visible is
# calling the same function twice: the first call leaves the list one
# reference short, and the second reads what it freed. So every function here
# is called more than once on the SAME structure, and the second answer has to
# equal the first.
from typing import Optional


class Node:
    v: int
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.nxt = None


def walk(head: Optional[Node], limit: Optional[int]) -> int:
    n = 0
    cur = head
    while cur is not None:
        if limit is not None and n >= limit:
            break
        n += 1
        cur = cur.nxt
    return n


def total(head: Optional[Node], stop_at: int) -> int:
    acc = 0
    cur = head
    while cur is not None:
        if cur.v == stop_at:
            break
        acc += cur.v
        cur = cur.nxt
    return acc


a = Node(1)
b = Node(2)
c = Node(3)
a.nxt = b
b.nxt = c

print(walk(a, None), walk(a, None))
print(walk(a, 2), walk(a, 2), walk(a, 0))
print(total(a, 3), total(a, 3), total(a, 1), total(a, 99))
print(a.v, b.v, c.v)
