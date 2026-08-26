# probe: leak -- a chain BUILT in a loop and then walked (20000 iterations)
# axes: op=leak-loop iterations=20000
#
# `cur = fresh` inside the loop is the other half of the linked-structure
# idiom: the merge argument is fed a fresh instance on the back edge and the
# pre-loop head on the entry edge, so the entry edge is the one that has to be
# lent a reference. Each node carries a list so one leaked node is over this
# instrument's 500 B floor -- see the note on leak_optionalwalk_loop.
#
# CPython 3.14 expects: 200000

from typing import Optional


class Node:
    v: int
    payload: list[int]
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.payload = [v, v, v, v, v, v, v, v]
        self.nxt = None


def build(n: int) -> Node:
    head = Node(0)
    cur = head
    i = 1
    while i < n:
        fresh = Node(i)
        cur.nxt = fresh
        cur = fresh
        i += 1
    return head


def once() -> int:
    head = build(5)
    acc = 0
    walk: Optional[Node] = head
    while walk is not None:
        acc += walk.v
        walk = walk.nxt
    return acc


total = 0
for _ in range(20000):
    total += once()
print(total)
