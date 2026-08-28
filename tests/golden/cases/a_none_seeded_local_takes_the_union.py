# WHAT: `best = None` at the head of an accumulator -- the shape every
# "smallest so far" / "first seen" loop is written in -- and the reads that
# follow it, guarded the way Python guards them.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answer is the value the
# accumulator holds at the end, and the guard that makes each read safe is a
# runtime test. A name that took the wrong union member would still compile.
#
# ⛔ A BINDING THAT READS THE NAME IS SKIPPED. `acc = acc + v` is a rebinding,
# and inferring it with the name still at None answers `object` and throws the
# whole seed away -- which is the running-total idiom, below.
def largest(xs: "list[int]") -> int:
    best = None
    for v in xs:
        if best is None or v > best:
            best = v
    if best is None:
        raise ValueError("empty")
    return best


print(largest([3, 1, 5, 2]))
try:
    print(largest([]))
except ValueError as e:
    print("ValueError:", e)


def running_total(xs: "list[int]") -> int:
    acc = None
    for v in xs:
        if acc is None:
            acc = v
        else:
            acc = acc + v
    if acc is None:
        return 0
    return acc


print(running_total([1, 2, 3]), running_total([]))


def label(flag: int) -> str:
    found = None
    if flag == 1:
        found = "text"
    elif flag == 2:
        found = 5
    if found is None:
        return "none"
    if isinstance(found, str):
        return "str:" + found
    return "int:" + str(found)


print(label(0), label(1), label(2))


class Item:
    name: str
    weight: int

    def __init__(self, name: str, weight: int) -> None:
        self.name = name
        self.weight = weight


def heaviest(items: "list[Item]") -> str:
    best = None
    for item in items:
        if best is None or item.weight > best.weight:
            best = item
    if best is None:
        return "-"
    return best.name


print(heaviest([Item("a", 2), Item("b", 5), Item("c", 1)]), heaviest([]))


def rebound() -> int:
    x = None
    x = 5
    return x + 1


print(rebound())


def never_rebound() -> str:
    x = None
    if x is None:
        return "still none"
    return "other"


print(never_rebound())
