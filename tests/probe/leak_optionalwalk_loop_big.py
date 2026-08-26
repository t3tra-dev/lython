# probe: leak -- a linked list walked with `while cur is not None` (20000 iterations)
# axes: op=leak-loop iterations=20000
# CPython 3.14 expects: 60000

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
for _ in range(20000):
    total += once()
print(total)
