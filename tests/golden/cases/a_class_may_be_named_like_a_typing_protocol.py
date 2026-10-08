# WHAT: program classes named Iterator, Sequence, Iterable and Sized -- names a
# typing protocol also has -- are those classes in annotations, containers and
# calls, while every list, dict, comprehension, generator and for loop in the
# same program keeps meaning what the protocols mean to them. An explicit
# `typing.Iterator[int]` still names the protocol.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the classes' fields and the
# builtins' results are run time values; the class used to be read as the
# protocol (refused), and once it was not, displacing the protocol's entry broke
# list methods and comprehensions in the same program.
import typing


class Iterator:
    def __init__(self, start: int) -> None:
        self.at = start

    def step(self) -> int:
        self.at += 1
        return self.at


class Sequence:
    def __init__(self, items: list[int]) -> None:
        self.items = items


class Iterable:
    def __init__(self) -> None:
        self.n = 3


class Sized:
    def __init__(self) -> None:
        self.size = 7


def advance(it: Iterator, times: int) -> int:
    for _ in range(times):
        it.step()
    return it.at


def total(s: Sequence) -> int:
    return sum(x for x in s.items)


def count_up(n: int) -> typing.Iterator[int]:
    i = 0
    while i < n:
        yield i
        i += 1


its: list[Iterator] = [Iterator(0), Iterator(10)]
print([advance(it, 3) for it in its])
print(total(Sequence([1, 2, 3])), Iterable().n, Sized().size)
xs = [3, 1, 2]
d = {"a": 1, "b": 2}
print(xs.index(1), sorted(xs), list(reversed(xs)), [x * 2 for x in xs])
print(list(d.keys()), list(d.items()), "a" in d, len(d))
print(list(count_up(4)), tuple(xs)[1:], {1, 2} | {3})
