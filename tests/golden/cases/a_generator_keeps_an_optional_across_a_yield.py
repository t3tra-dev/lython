# Optional values crossing a generator's suspension: a union parameter, a
# union local live across yields, a union yielded, and a generator of a
# member type read where the union is expected. A union crosses the resume
# boxed, and the printed values are what comes back out of the box.
from typing import Iterator


class Node:
    def __init__(self, name: str, nxt: "Node | None") -> None:
        self.name = name
        self.nxt = nxt


def walk(head: Node | None) -> Iterator[str]:
    cur = head
    last: str | None = None
    while cur is not None:
        yield cur.name
        last = cur.name
        cur = cur.nxt
    if last is not None:
        yield "last=" + last


def maybe_names(n: int) -> Iterator[str | None]:
    for i in range(n):
        yield ("n" + str(i)) if i % 2 == 0 else None


def letters(text: str) -> Iterator[str]:
    for c in text:
        yield c


def show(it: Iterator[str | None]) -> None:
    for v in it:
        print(v)


def pairwise(xs: list[int]) -> Iterator[int]:
    prev: int | None = None
    for x in xs:
        if prev is not None:
            yield prev + x
        prev = x


chain = Node("a", Node("b", Node("c", None)))
print(list(walk(chain)))
print(list(walk(None)))
show(maybe_names(3))
show(letters("xy"))
print(list(pairwise([1, 2, 3])))
