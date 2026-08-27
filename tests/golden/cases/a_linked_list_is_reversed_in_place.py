# WHAT: the linked-list reversal -- `nxt = cur.nxt; cur.nxt = prev; prev = cur;
# cur = nxt` -- where TWO Optional locals are carried across one loop and each
# takes over what the other lets go on the same edge.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: a reversal is only correct
# if every node kept exactly the reference it had, and neither failure mode
# says so. One reference too few frees a node the caller still holds, one too
# many keeps a chain alive that CPython would have dropped, and either way the
# program keeps running. So the assertions walk the reversed chain end to end,
# read the old head from the caller's side afterwards, and reverse the result a
# second time to get the original order back.
from typing import Optional


class Node:
    v: int
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.nxt = None


def chain(n: int) -> Node:
    head = Node(0)
    cur = head
    i = 1
    while i < n:
        fresh = Node(i)
        cur.nxt = fresh
        cur = fresh
        i += 1
    return head


def reverse(head: Node) -> Node:
    prev: Optional[Node] = None
    cur: Optional[Node] = head
    while cur is not None:
        nxt = cur.nxt
        cur.nxt = prev
        prev = cur
        cur = nxt
    if prev is None:
        return head
    return prev


def spell(head: Optional[Node]) -> str:
    out = ""
    cur = head
    while cur is not None:
        out += str(cur.v)
        cur = cur.nxt
    return out


for size in [1, 2, 5]:
    built = chain(size)
    forward = spell(built)
    reversed_once = reverse(built)
    print(size, forward, spell(reversed_once), built.v, built.nxt is None)
    back = reverse(reversed_once)
    print(spell(back), back.v)

# The old head is the new tail, and it is still the caller's to read.
a = chain(3)
r = reverse(a)
print(spell(r), a.v, a.nxt is None, r.v)
