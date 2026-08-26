# WHAT: a function that returns `T | None` used as a loop's condition and its
# own next value -- `while cur is not None: cur = step(cur)` -- which is how a
# search or a cursor is written when the end is a value rather than a length.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the loop carries a TAG and
# a payload across the back edge, and both are renamed by the edge. What the
# printed numbers decide is whether the arm the tag selects is the arm the body
# read: a walk that lost the tag stops at the wrong iteration or reads a
# payload that is not there, and neither shows up as a refusal.
from typing import Optional


class Node:
    v: int

    def __init__(self, v: int) -> None:
        self.v = v


def step(i: int) -> Optional[Node]:
    if i < 4:
        return Node(i)
    return None


total = 0
seen = 0
cur: Optional[Node] = step(0)
while cur is not None:
    total += cur.v
    seen += 1
    cur = step(cur.v + 1)
print(total, seen, cur is None)


def count_down(start: int) -> int:
    n = 0
    node: Optional[Node] = step(start)
    while node is not None:
        n += node.v * 100
        node = step(node.v + 1)
    return n


print(count_down(0), count_down(2), count_down(9))
