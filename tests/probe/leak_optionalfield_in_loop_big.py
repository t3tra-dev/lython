# probe: leak -- an Optional value written into an Optional FIELD inside a loop (20000 iterations)
# axes: op=leak-loop iterations=20000
#
# The store used to be an `scf.if` on the tag; it is one `select` on the entity
# word now, and the arm that is absent relies on the immortal dead placeholder
# being retained and released for free. That is a retain and a release moved out
# from under a branch, so this weighs whether they still pair.
#
# CPython 3.14 expects: 60000

from typing import Optional


class Node:
    v: int
    payload: list[int]
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.payload = [v] * 16
        self.nxt = None


def relink(head: Node, tail: Optional[Node]) -> int:
    seen = 0
    cur: Optional[Node] = head
    while cur is not None:
        seen += 1
        if cur.v == 0:
            cur.nxt = tail
        cur = cur.nxt
    return seen


def once() -> int:
    a = Node(0)
    b = Node(1)
    c = Node(2)
    b.nxt = c
    total = relink(a, b)
    total += relink(a, None)
    return total


total = 0
for _ in range(20000):
    total += once()
print(total)
