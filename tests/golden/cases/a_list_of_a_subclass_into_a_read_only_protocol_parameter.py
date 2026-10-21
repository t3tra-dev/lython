# WHAT: a list of a subclass handed to a constructor whose parameter is
# `Iterable[Case | Group]`, the items then held as that union and each asked
# for its name -- the subclass's override, a nested group, and the base.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the argument was refused
# at an inlined constructor while a free function took it; that each element
# kept its own class through the union is only seen in the names it answers.
from collections.abc import Iterable


class Case:
    def __init__(self, n: str) -> None:
        self.n = n

    def name(self) -> str:
        return "case " + self.n


class Group:
    def __init__(self, items: Iterable[Case | Group] = ()) -> None:
        self.items: list[Case | Group] = []
        for item in items:
            self.items.append(item)

    def name(self) -> str:
        return "group of %d" % len(self.items)

    def names(self) -> list[str]:
        out: list[str] = []
        for item in self.items:
            if isinstance(item, Case):
                out.append(item.name())
            else:
                out.append(item.name())
        return out


class Mine(Case):
    def name(self) -> str:
        return "mine " + self.n


inner = Group([Mine("a"), Mine("b")])
print(inner.names())
print(Group([inner]).names(), Group([Case("c")]).names(), Group().names())
