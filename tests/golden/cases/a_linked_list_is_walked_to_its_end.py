# WHAT: the loop every linked structure is read with -- `while cur is not None:
# cur = cur.nxt` -- over a borrowed parameter, over a module-level name, and
# over a list built in the same function.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: what the walk carries
# across the back edge is a reference, and the question is whether the frame
# holds one on every iteration. A wrong answer is not a refusal, it is a sum
# taken from a node that was freed or a loop that stops at the wrong link, so
# the totals and the final `is None` are the assertions. `sum_from` takes its
# head BORROWED, which is the accounting the walk has to get right: the first
# iteration releases the caller's node under the loop's own name.
from typing import Optional


class Node:
    v: int
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.nxt = None


def chain(values: list[int]) -> Node:
    head = Node(values[0])
    head.nxt = Node(values[1])
    tail = head.nxt
    if tail is not None:
        tail.nxt = Node(values[2])
    return head


def sum_from(head: Node) -> int:
    acc = 0
    cur: Optional[Node] = head
    while cur is not None:
        acc += cur.v
        cur = cur.nxt
    return acc


def length_of(head: Node) -> int:
    n = 0
    cur: Optional[Node] = head
    while cur is not None:
        n += 1
        cur = cur.nxt
    return n


first = chain([1, 2, 3])
print(sum_from(first), length_of(first))

# The same walk at module level, and the head is still usable afterwards.
walk: Optional[Node] = first
total = 0
while walk is not None:
    total += walk.v * 10
    walk = walk.nxt
print(total, walk is None, first.v)

# A single node walks once and stops.
alone = Node(7)
print(sum_from(alone), length_of(alone), alone.nxt is None)
