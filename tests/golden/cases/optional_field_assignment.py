# Why execution: assigning to an `Optional[T]` field was refused outright
# ("attribute value ABI has 2 values, but storage expects 3"), so nothing here
# ran at all. Every store below is one the compiler used to reject, and the
# values are the assertion that the union's tag and its payload agree
# afterwards -- a store that lands on the wrong lane reads back as the wrong
# member, which only running shows.
from typing import Optional


class Node:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tag: Optional[str] = None
        self.rank: Optional[int] = None


a = Node("a")
print(a.name, a.tag is None, a.rank is None)

# A concrete value into each member type.
a.tag = "set"
print(a.tag is None, a.rank is None)
a.rank = 7
print(a.tag is None, a.rank is None)

# Back to None, which is the assignment the type check used to refuse on its
# own: `Optional` carries None as a LITERAL member while a `None` expression
# arrives as the contract, and only one direction of that pair was spelled.
a.tag = None
print(a.tag is None, a.rank is None)
a.rank = None
print(a.tag is None, a.rank is None)

# Repeated stores over one slot: the old member's reference is released once
# and the new one retained once, whichever members they are.
b = Node("b")
b.tag = "one"
b.tag = "two"
b.tag = None
b.tag = "three"
print(b.name, b.tag is None, b.rank is None)

# Stored from a parameter rather than a literal, and from inside a method.
class Holder:
    def __init__(self) -> None:
        self.slot: Optional[str] = None

    def put(self, v: str) -> None:
        self.slot = v

    def clear(self) -> None:
        self.slot = None


h = Holder()
print(h.slot is None)
h.put("x")
print(h.slot is None)
h.clear()
print(h.slot is None)
