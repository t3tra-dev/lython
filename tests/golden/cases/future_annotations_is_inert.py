# `from __future__ import annotations` was refused as "unsupported import
# '__future__.annotations'", which took the whole file with it -- and the
# feature it names asks that annotations be treated as strings, which is what
# this compiler does with them anyway. Must run: what regresses is the refusal,
# but only the values show the annotations still RESOLVED after the import was
# accepted; binding nothing and typing everything as object would compile the
# import just as quietly.
from __future__ import annotations


class Node:
    def __init__(self, v: int) -> None:
        self.v = v

    # Under PEP 563 this is a forward reference to the class being defined.
    def twin(self) -> Node:
        return Node(self.v)

    def pair(self) -> tuple[Node, int]:
        return (self.twin(), self.v)


def count(xs: list[int]) -> int:
    return len(xs)


def totals(d: dict[str, list[int]]) -> int:
    t = 0
    for name in sorted(d):
        for v in d[name]:
            t = t + v
    return t


n = Node(3)
print(n.twin().v, n.pair()[1])
print(count([1, 2]), totals({"a": [1, 2], "b": [3]}))

nodes: list[Node] = [Node(1), Node(2)]
total = 0
for item in nodes:
    total = total + item.v
print(total)
