# probe: leak -- the carried Optional is READ after the loop (100 iterations)
# axes: op=leak-loop iterations=100
#
# The loop's exit release is now emitted only when the entry edge acquired
# something, which is a release TAKEN AWAY -- so this weighs the shape that
# release used to cover. Both spellings are here: a frame-owned initial value
# (where the retain is not taken) and a borrowed parameter (where it is), since
# the two go down opposite sides of that gate.
#
# CPython 3.14 expects: 500

from typing import Optional


class Node:
    v: int
    payload: list[int]
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.payload = [v, v, v, v, v, v, v, v]
        self.nxt = None


def from_parameter(head: Node) -> int:
    cur: Optional[Node] = head
    while cur is not None and cur.v < 2:
        cur = cur.nxt
    if cur is not None:
        return cur.v
    return -1


def once() -> int:
    a = Node(1)
    b = Node(2)
    c = Node(3)
    a.nxt = b
    b.nxt = c
    n: Optional[Node] = a
    while n is not None and n.v < 3:
        n = n.nxt
    seen = 0
    if n is not None:
        seen = n.v
    return seen + from_parameter(a)


total = 0
for _ in range(100):
    total += once()
print(total)
