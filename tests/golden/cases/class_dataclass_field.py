# What: dataclasses.field() static subset — default= feeds the plain
# default channel (and the class attribute), default_factory= re-evaluates
# per omitted argument, and an explicit argument suppresses the factory.
from dataclasses import dataclass, field

counter: int = 0


def next_id() -> int:
    global counter
    counter += 1
    return counter


def make_tag() -> str:
    return "fresh"


@dataclass
class Node:
    label: str
    tag: str = field(default="t0")
    ident: int = field(default_factory=next_id)
    note: str = field(default_factory=make_tag)


n1 = Node("a")
n2 = Node("b")
n3 = Node("c", "t9", 99, "n9")
print(n1.ident, n2.ident, n3.ident)
print(n1.tag, n3.tag)
print(n1.note, n3.note)
print(repr(n2))
print(n1 == n1, n1 == n2)
print(Node.tag)
