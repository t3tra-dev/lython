# WHAT: a parameter annotated with a protocol -- Container, Collection,
# Iterable, Sized, Sequence -- holds the very list, tuple, dict, set, range or
# str its call gave: membership, length, iteration and indexing answer as that
# value does, in a free function, a method and a constructor; a str read
# where the protocol says `object` is still the str.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answers are the
# values' own. `2 in c` for a list or a tuple crashed at run time -- the call
# boxed the list as an `object` and the function read the box as the list --
# and a str iterated or indexed as `Iterable[object]` / `Sequence[object]`
# was refused in the lowering.
from typing import Collection, Container, Iterable, Sequence, Sized


def has(c: Container[int]) -> bool:
    return 2 in c


print(has([1, 2]), has([1, 3]), has((2, 3)), has((3,)), has({2: "x"}),
      has({3: "x"}), has({1, 2}), has(range(3)), has(range(3, 5)))


def summary(c: Collection[int]) -> str:
    total = 0
    for x in c:
        total += x
    return str(len(c)) + ":" + str(total) + ":" + str(5 in c)


print(summary([1, 2]), summary((5, 6, 7)), summary({4, 5}), summary(range(4)))


def shown(c: Iterable[object]) -> list[str]:
    out: list[str] = []
    for x in c:
        out.append(repr(x))
    return out


print(shown("ab"), shown([1, "s", None]), shown({"k": 2}), shown(range(2)))


def at(a: Sequence[object], i: int) -> str:
    return repr(a[i])


print(at("abc", 1), at("abc", -1), at([1, 2], 0), at((None, 3.5), 1),
      at(b"xy", 0), at(range(5), 3))


def size(c: Sized) -> int:
    return len(c)


print(size([1]), size("abc"), size({}), size(range(10)))


class Bag:
    def __init__(self, items: Iterable[int]) -> None:
        self.items: list[int] = []
        for x in items:
            self.items.append(x)

    def common(self, other: Container[int]) -> list[int]:
        return [x for x in self.items if x in other]


bag = Bag(range(5))
print(bag.items, Bag((7, 8)).items, Bag({9}).items)
print(bag.common([1, 3, 9]), bag.common((4,)), bag.common({0: "z"}))


class Checks:
    def assertIn[T](self, member: T, container: Container[T]) -> str:
        return "in" if member in container else "missing"


print(Checks().assertIn(1, [1, 2]), Checks().assertIn("z", "abc"),
      Checks().assertIn("k", {"k": 1}))
