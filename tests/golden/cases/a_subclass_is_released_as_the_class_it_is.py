# Why execution: the leak gate measures this program (tests/CMakeLists.txt),
# and only a run shows what a release leaves allocated.
#
# An instance released where its static type is a base -- returned as the
# base, returned as a union of the base, yielded from a union -- went through
# the base's deallocator, which knows only the base's fields: the subclass's
# list was never released (64 B per call). A grandchild is released as
# itself too, past a child that adds no field.
from typing import Iterator


class Entry:
    def __init__(self, key: str) -> None:
        self.key = key


class BigEntry(Entry):
    def __init__(self, key: str, extra: list[int]) -> None:
        super().__init__(key)
        self.extra = extra


class Middle(Entry):
    pass


class Leaf(Middle):
    def __init__(self, key: str, tags: list[str]) -> None:
        super().__init__(key)
        self.tags = tags


class Other:
    def __init__(self, n: int) -> None:
        self.n = n


def as_base(i: int) -> Entry:
    if i % 3 == 0:
        return Leaf("l", ["x"] * i)
    return BigEntry("b", [i] * 8)


def as_union(i: int) -> Entry | Other:
    if i % 2 == 0:
        return as_base(i)
    return Other(i)


def pairs(n: int) -> Iterator[Entry | Other]:
    for i in range(n):
        x: Entry | Other = as_union(i)
        yield x


total = 0
for i in range(60):
    e = as_base(i)
    total += len(e.key)
    u = as_union(i)
    total += u.n if isinstance(u, Other) else len(u.key)
for p in pairs(40):
    total += 1 if isinstance(p, Entry) else p.n
print(total, type(as_base(3)).__name__, type(as_union(2)).__name__)
