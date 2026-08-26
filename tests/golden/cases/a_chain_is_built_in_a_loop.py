# WHAT: `cur = fresh` -- the loop that BUILDS a linked structure, where the
# merge argument is fed the pre-loop head on one edge and a fresh instance on
# the other. With the walk that reads one, it is the whole idiom.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: two of the three things
# that were wrong here have no diagnostic. The lend the entry edge needs was
# refused outright (that one does), but writing it at the raw allocation and
# reading the field-evidence cache through the merge argument are a wrong
# refcount and a wrong ARM -- the last node's `nxt` was never written, and a
# read that believed the cache followed a null handle. So the assertions are
# the values, and the last one on every chain is the empty link.
from typing import Optional


class Node:
    v: int
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
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


def total(head: Node) -> int:
    acc = 0
    cur: Optional[Node] = head
    while cur is not None:
        acc += cur.v
        cur = cur.nxt
    return acc


def length(head: Node) -> int:
    n = 0
    cur: Optional[Node] = head
    while cur is not None:
        n += 1
        cur = cur.nxt
    return n


for size in [1, 2, 5]:
    built = build(size)
    print(size, total(built), length(built), built.v, built.nxt is None)

# The same loop at module level, and the tail's own link is the empty one.
head = Node(100)
tail = head
k = 1
while k < 4:
    node = Node(100 + k)
    tail.nxt = node
    tail = node
    k += 1
print(total(head), length(head), head.v, tail.v, tail.nxt is None, head.nxt is None)
