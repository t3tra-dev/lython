# probe: two Optional locals carried across one loop, cross-assigned -- the
# linked-list reversal
# axes: acquire=param width=optional op=carry flow=loop observe=writeback
# CLASSIFICATION @ 2026-08-27: 1 正しい
#
# FIXED. Three separate things were wrong with the reversal and each hid the
# next; this file kept the shape while they were found.
#
#   the field store was an `scf.if` on the tag, and the release planner's
#   liveness walk gives up on a group whose use sits in a nested region;
#
#   the loop's entry-edge lend seeded no merge candidate, because it is a
#   `py.incref` the emitter wrote -- a call with no results and no marker --
#   so nothing discharged it where the body never ran;
#
#   and the borrowed-entry walk credited a release through a name the back
#   edge had REBOUND. `prev = cur; cur = nxt` renames the group from the cur
#   lane onto the prev lane and rebinds the cur lane on that same edge, so the
#   loop-exit release of the cur lane was read as the return of the
#   parameter's lend -- and the prev lane's own release then found a balance
#   of zero. That last one is why the IR balanced by hand while the walk
#   refused: nothing was wrong with what was emitted.
#
# tests/golden/cases/a_linked_list_is_reversed_in_place.py runs the shape and
# reverses twice to get the original order back;
# tests/probe/leak_reverse_two_lanes_* weighs it at 60 B per iteration over
# 20000 reversals of a five-node chain, each node carrying a list.
#
# CPython 3.14 expects: 2 0

from typing import Optional


class Node:
    v: int
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.nxt = None


def rev(head: Node) -> Node:
    prev: Optional[Node] = None
    cur: Optional[Node] = head
    while cur is not None:
        nxt = cur.nxt
        cur.nxt = prev
        prev = cur
        cur = nxt
    if prev is None:
        return head
    return prev


a = Node(0)
b = Node(1)
c = Node(2)
a.nxt = b
b.nxt = c
r = rev(a)
print(r.v, a.v)
