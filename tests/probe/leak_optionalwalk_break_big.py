# probe: leak -- `break` out of a loop carrying an Optional (20000 iterations)
# axes: op=leak-loop iterations=20000
#
# The repair this weighs removed a release from the break edge: the edge's
# re-wrap of a narrowed name had read as a REPLACEMENT, so the token was spent
# twice. Taking a release away is how a leak is introduced, which is what this
# is here to say did not happen. Each node carries a list so one leaked node is
# worth more than this instrument's floor.
#
# CPython 3.14 expects: 40000

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
    n = 0
    cur: Optional[Node] = a
    while cur is not None:
        if n >= 2:
            break
        n += 1
        cur = cur.nxt
    return n


total = 0
for _ in range(20000):
    total += once()
print(total)
