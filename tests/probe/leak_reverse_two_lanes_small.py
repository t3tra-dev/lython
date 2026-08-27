# probe: leak -- a chain reversed in place, two Optional lanes cross-assigned (100 iterations)
# axes: op=leak-loop iterations=100
#
# Each edge hands one lane's token to the other rather than releasing and
# retaining, and the borrowed-entry walk was taught to stop crediting a release
# through a name the edge rebinds. Neither is a change to what is EMITTED, so
# this weighs whether the IR that was always there actually balances.
#
# CPython 3.14 expects: 1000

from typing import Optional


class Node:
    v: int
    payload: list[int]
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.payload = [v] * 16
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


def once() -> int:
    built = chain(5)
    r = reverse(built)
    total = 0
    walk: Optional[Node] = r
    while walk is not None:
        total += walk.v
        walk = walk.nxt
    return total


total = 0
for _ in range(100):
    total += once()
print(total)
