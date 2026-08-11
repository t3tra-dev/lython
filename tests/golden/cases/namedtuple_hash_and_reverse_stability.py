# Why execution: both are values only running produces.
#
#   - a NamedTuple compares equal to a plain tuple with the same contents, so
#     it must HASH equal too. The synthesized __hash__ was an XOR fold of the
#     fields, which happens to satisfy "equal objects hash equal" only among
#     NamedTuples -- a dict keyed by tuples then missed a NamedTuple key that
#     compares equal to one already in it.
#   - sorted(reverse=True) must keep equal elements in their ORIGINAL order.
#     The decorate-sort-undecorate breaks ties with an index, and that only
#     helps when two keys compare EQUAL: a class whose __lt__ answers False
#     both ways is neither less nor equal, so the sort left the pair alone and
#     the backward undecorate swapped it.
from typing import NamedTuple


class Point(NamedTuple):
    x: int


class Pair(NamedTuple):
    a: int
    b: str


class Unordered:
    def __init__(self, name: str) -> None:
        self.name: str = name

    def __lt__(self, other: "Unordered") -> bool:
        return False


def namedtuple_hashes_as_its_tuple() -> None:
    print(hash(Point(3)) == hash((3,)))
    print(hash(Pair(1, "a")) == hash((1, "a")))
    print(hash(Point(0)) == hash((0,)), hash(Point(-1)) == hash((-1,)))


def namedtuple_as_a_tuple_key() -> None:
    d: dict[tuple[int, str], str] = {}
    d[(1, "a")] = "from tuple"
    print(d[(1, "a")])


def reverse_keeps_original_order() -> None:
    print(sorted([Unordered("a"), Unordered("b")], reverse=True)[0].name)
    print(sorted([Unordered("a"), Unordered("b")])[0].name)
    three = [Unordered("a"), Unordered("b"), Unordered("c")]
    print(sorted(three, reverse=True)[0].name, sorted(three, reverse=True)[2].name)


def reverse_still_orders_what_it_can() -> None:
    xs = [3, 1, 3, 1]
    print(sorted(xs, reverse=True), sorted(xs))
    ys = ["bb", "a", "cc", "d"]
    print(sorted(ys, key=len, reverse=True))
    pairs = [(1, "a"), (0, "b"), (1, "c"), (0, "d")]
    print(sorted(pairs, reverse=True))


def main() -> None:
    namedtuple_hashes_as_its_tuple()
    namedtuple_as_a_tuple_key()
    reverse_keeps_original_order()
    reverse_still_orders_what_it_can()


main()
