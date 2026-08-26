# probe: two Optional locals carried across one loop, cross-assigned -- the
# linked-list reversal
# axes: acquire=param width=optional op=carry flow=loop observe=refusal
# CLASSIFICATION @ 2026-08-27: 3 loud 拒否 (診断)
#   owned resource from builtin.unrealized_conversion_cast result 0 reaches
#   function exit without release, transfer, or owned return
#
# The reduced spellings below report the other half of the same imbalance,
# "borrowed entry argument 0 of @rev is released or transferred without a
# prior retain", which is what says the two lanes disagree about who holds it.
#
# The walk (one carried Optional) and the build (`cur = fresh`) both run now;
# this is the shape where TWO carried lanes name one object for part of an
# iteration. `prev = cur` makes the prev lane acquire what the cur lane is
# abandoning on the same edge, which `carriedLoopEdgeOperands` handles by
# handing the token over rather than releasing and retaining -- and the borrowed
# walk still reports the parameter released without its retain.
#
# ⛔ NOT the `cur.nxt = prev` store: dropping it leaves the same refusal, so it
# is the two carried lanes and not the field write. Dropping `prev = cur`
# instead reports the OTHER end of the same imbalance -- "reaches function exit
# with 1 retained ownership token(s)", the loop's entry retain undischarged --
# so both halves of that contract are involved and neither is the whole story.
#
# ⛔ AND NOT reachable as a wrong answer: it is a refusal at the affine
# ownership verifier, before any code runs.
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
