# This crashed the COMPILER: SIGILL with zero stdout and zero stderr. A
# union-typed field is stored inline, so a class reachable from its own field
# through one has no finite layout and the expansion recursed until the stack
# ran out. `nxt: "Node"` is stored as a reference and works.
from typing import Optional


class Node:
    v: int
    nxt: Optional["Node"]

    def __init__(self, v: int) -> None:
        self.v = v
        self.nxt = None


print(Node(1).v)
