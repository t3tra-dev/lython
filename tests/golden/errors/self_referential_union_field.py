# This crashed the COMPILER: SIGILL with zero stdout and zero stderr. A
# union-typed field is stored inline, so a class reachable from its own field
# through one has no finite layout and the expansion recursed until the stack
# ran out. `nxt: "Node"` is stored as a reference and works.
#
# ⛔ AND `Optional["Node"]` IS HOW EVERY LINKED STRUCTURE IS SPELLED, so it is
# worth writing down what a repair needs rather than only what it refuses.
# Re-measured 2026-08-20: `nxt: "Node"` and `kids: list["Tree"]` both work, so
# the boundary is exactly the union.
#
# A field typed `C | None` with ONE class member has a faithful boxed form: the
# box holds C's payload, and None is the EMPTY box -- class id 0, entity 0,
# which the object path already reads as None. So `classFieldStoredBoxed` could
# say yes for that one union shape and the layout would terminate.
#
# What stops it is the LOAD: rebuilding a union value from the box needs a tag
# chosen at run time, and building a union at run time is the mechanism this
# compiler does not have (the same one `e.__cause__` waits on). Storing is the
# easy half. Until that exists, the refusal is the answer, and `nxt: "Node"`
# with a sentinel instance is the spelling that works.
from typing import Optional


class Node:
    v: int
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.nxt = None


print(Node(1).v)
