# Why execution: equality was already right -- Key(0) == Key(0) is True -- and
# the failure was a KeyError at runtime, because the dict probes by hash first
# and two equal keys landed in different buckets. Only running the lookup shows
# it.
#
# CPython's namedtuple inherits tuple.__hash__. A plain dataclass gets
# __hash__ = None instead, so it is deliberately absent there.
from typing import NamedTuple


class Key(NamedTuple):
    row: int


class Pair(NamedTuple):
    x: int
    y: str


class Fixed(NamedTuple):
    v: int

    def __hash__(self) -> int:
        return 42


def main() -> None:
    single: dict[Key, str] = {}
    single[Key(0)] = "origin"
    single[Key(1)] = "next"
    print(single[Key(0)])
    print(single[Key(1)])
    print(len(single))

    pairs: dict[Pair, int] = {}
    pairs[Pair(1, "a")] = 7
    print(pairs[Pair(1, "a")])

    seen: set[Key] = set()
    seen.add(Key(3))
    print(Key(3) in seen)
    print(Key(4) in seen)

    print(hash(Fixed(1)))
    print(Key(0) == Key(0))


main()
