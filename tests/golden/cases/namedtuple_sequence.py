# Why execution: the values are what the fold has to produce. A NamedTuple IS
# a tuple, so len(p) is its field count and p[0] is the field at that
# position -- both were "contract 'P' does not provide manifest method
# '__len__' / '__getitem__'" while p.x and print(p) worked.
#
# len() is synthesized (the count is known where the class is), and the
# subscript folds at the subscript, where the index is: a literal index names
# a field, a computed one would need a real tuple to index and the fields do
# not share a type anyway.
from typing import NamedTuple


class Point(NamedTuple):
    x: int
    y: int = 0


class Row(NamedTuple):
    name: str
    count: int


def main() -> None:
    p = Point(1, 2)
    print(p, p.x, p.y, len(p))
    print(p[0], p[1], p[-1], p[-2])
    q = Point(1)
    print(q, q[1], p == Point(1, 2), p == q)
    r = Row("a", 3)
    print(r, len(r), r[0], r[1])
    print({p: "here"}[Point(1, 2)])
    total = 0
    for i in range(len(p)):
        total += 1
    print(total)


main()
