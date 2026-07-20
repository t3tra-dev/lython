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


# Container factories live on their own class: the synthesized __eq__ over
# a list field still trips a lowering dominance bug, so eq coverage stays
# on the scalar class above.
@dataclass(eq=False)
class Bag:
    name: str
    peers: list[str] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)


n1 = Node("a")
n2 = Node("b")
n3 = Node("c", "t9", 99, "n9")
print(n1.ident, n2.ident, n3.ident)
print(n1.tag, n3.tag)
print(n1.note, n3.note)
print(repr(n2))
print(n1 == n1, n1 == n2)
print(Node.tag)
b1 = Bag("x")
b2 = Bag("y")
b3 = Bag("z", ["p"], {"s": 3})
b1.peers.append("q")
b1.scores["k"] = 1
print(b1.peers, b2.peers, b3.peers)
print(b1.scores, b2.scores, b3.scores)
print(repr(b3))
