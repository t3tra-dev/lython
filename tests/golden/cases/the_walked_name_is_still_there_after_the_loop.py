# WHAT: reading the name a `while cur is not None:` loop carried, AFTER the
# loop -- narrowed, in both the statement and the expression spelling -- and
# `A and B` as the loop's own test, which is how a bounded search is written.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the loop's exit release
# used to be written at the top of the after-block, before any of these reads.
# For a name the frame already owned that release was one too many, so the
# read that follows it read freed memory -- and what the reader gets back is a
# VALUE, not a diagnostic. Every line prints the field of the node the search
# stopped on, and the `None` outcome beside it.
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


def stop_at(head: Node, want: int) -> int:
    cur: Optional[Node] = head
    while cur is not None and cur.v < want:
        cur = cur.nxt
    if cur is not None:
        return cur.v
    return -1


print(stop_at(chain(4), 0), stop_at(chain(4), 2), stop_at(chain(4), 9))

# The same at module level, where the initial value is a name the frame owns.
head = chain(4)
n: Optional[Node] = head
while n is not None and n.v < 2:
    n = n.nxt
print(n is None)
if n is not None:
    print(n.v, n.nxt is None)
print(n.v if n is not None else -1, head.v)

# And a walk that runs off the end: the name is None afterwards and stays usable.
m: Optional[Node] = head
while m is not None:
    m = m.nxt
print(m is None, m.v if m is not None else -1, head.v)
