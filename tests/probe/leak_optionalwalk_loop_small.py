# probe: leak -- a linked list walked with `while cur is not None` (100 iterations)
# axes: op=leak-loop iterations=100
# CPython 3.14 expects: 300

from typing import Optional


class Node:
    v: int
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.nxt = None


def once() -> int:
    a = Node(1)
    b = Node(2)
    a.nxt = b
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
