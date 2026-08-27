# probe: two Optional locals carried across one loop, cross-assigned -- the
# linked-list reversal
# axes: acquire=param width=optional op=carry flow=loop observe=refusal
# CLASSIFICATION @ 2026-08-27: 3 loud 拒否 (診断)
#   borrowed entry argument 0 of @rev is returned as owned without a
#   dominating retain
#
# The spelling that returns an int rather than the node reports the other half,
# "released or transferred without a prior retain". Both are the borrowed-entry
# WALK, not the release placement.
#
# The walk (one carried Optional) and the build (`cur = fresh`) both run now;
# this is the shape where TWO carried lanes name one object for part of an
# iteration.
#
# ⛔ NOT the `cur.nxt = prev` store, and not the borrowed initial value. Both
# were fixed 2026-08-27 and are covered by
# tests/golden/cases/a_loop_over_a_borrowed_optional_writes_its_field.py: the
# store is one `select` on the entity word rather than an `scf.if` the release
# planner will not place into, and the emitter's entry-edge lend now seeds a
# merge candidate so this pass discharges it. Dropping `prev = cur` from the
# body -- one carried lane instead of two -- compiles and matches CPython.
#
# What is left is the CROSS-ASSIGNMENT: `prev = cur` makes the prev lane
# acquire on the same edge that the cur lane abandons, so one object is named
# by two carried lanes for part of an iteration.
# `carriedLoopEdgeOperands` hands the token over rather than releasing and
# retaining (`transferred`), and the emitted IR balances -- counted by hand on
# a one-node chain: one `py.incref` on the parameter, one release of it at the
# loop's exit. It is the borrowed-entry WALK that refuses, so the next attempt
# should start by asking which path it takes to reach a release with
# `retained == 0`, not by re-reading the ledger.
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
