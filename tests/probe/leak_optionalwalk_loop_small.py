# probe: leak -- a linked list walked with `while cur is not None` (100 iterations)
# axes: op=leak-loop iterations=100
#
# ⛔ EACH NODE CARRIES A LIST, and the first version of this probe did not. A
# leaked bare instance is 80 bytes and a two-node list leaks 160 per call, an
# order of magnitude under this instrument's 500 B floor -- so the probe
# reported "no leak" over a walk that was leaking its whole chain, and the
# defect shipped. With a list on each node the same shape measures 352 B for
# two nodes and 2443 for nine, and the slope is what says one node per link.
#
# CPython 3.14 expects: 600

from typing import Optional


class Node:
    v: int
    payload: list[int]
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.payload = [v, v, v, v, v, v, v, v]
        self.nxt = None


def once() -> int:
    a = Node(1)
    b = Node(2)
    c = Node(3)
    a.nxt = b
    b.nxt = c
    acc = 0
    cur: Optional[Node] = a
    while cur is not None:
        acc += cur.v
        cur = cur.nxt
    return acc


total = 0
for _ in range(100):
    total += once()
print(total)
