# probe: leak -- a loop driven by a function returning `T | None` (20000 iterations)
# axes: op=leak-loop iterations=20000
#
# Each trip allocates a Node, carries it across the back edge as the payload of
# a union, and drops the previous one. Both the tag and the payload are renamed
# by the edge, and the release that balances the call's +1 has to land on the
# name the edge produced.
#
# The node carries a LIST so that dropping one is worth measuring: four bare
# 88-byte instances per iteration is under this instrument's 500 B floor, and
# the same shape would report "no leak" whether or not one is there.
#
# CPython 3.14 expects: 120000

from typing import Optional


class Node:
    v: int
    payload: list[int]

    def __init__(self, v: int) -> None:
        self.v = v
        self.payload = [v, v, v, v, v, v, v, v]


def step(i: int) -> Optional[Node]:
    if i < 4:
        return Node(i)
    return None


def once() -> int:
    total = 0
    cur: Optional[Node] = step(0)
    while cur is not None:
        total += cur.v
        cur = step(cur.v + 1)
    return total


total = 0
for _ in range(20000):
    total += once()
print(total)
