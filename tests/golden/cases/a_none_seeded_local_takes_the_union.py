# WHAT: `best = None` at the head of an accumulator -- the shape every
# "smallest so far" / "first seen" loop is written in -- and the reads that
# follow it, guarded the way Python guards them.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answer is the value the
# accumulator holds at the end, and the guard that makes each read safe is a
# runtime test. A name that took the wrong union member would still compile.
#
# ⛔ THE GUARD AROUND THE BINDING IS PART OF THE ANSWER. The walk of an
# optional linked structure binds the accumulator from `cur.value` inside
# `while cur is not None`, so the scan has to carry that narrowing with it --
# reading the same statement outside the guard infers an attribute of a union
# and abandons the seed for a program the guard makes exact.
#
# ⛔ AND SO IS A SECOND `= None`. Resetting the accumulator is how a scan
# starts the next group, and reading that binding as a type gave up on the
# whole seed -- for the shape the seed is most needed in. The reset also has to
# KEEP the union rather than pin the name back to None, which is what the
# annotated spelling has always done.
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


class Node:
    value: int
    nxt: "Node | None"

    def __init__(self, value: int) -> None:
        self.value = value
        self.nxt = None


def largest_in_chain(head: "Node | None") -> int:
    best = None
    cur = head
    while cur is not None:
        if best is None or cur.value > best:
            best = cur.value
        cur = cur.nxt
    if best is None:
        return 0
    return best


a = Node(3)
b = Node(7)
c = Node(5)
a.nxt = b
b.nxt = c
print(largest_in_chain(a), largest_in_chain(None))


def group_words(src: str) -> "list[str]":
    out: "list[str]" = []
    cur = None
    for ch in src:
        if ch == " ":
            if cur is not None:
                out.append(cur)
                cur = None
        elif cur is None:
            cur = ch
        else:
            cur = cur + ch
    if cur is not None:
        out.append(cur)
    return out


print(group_words("ab cd  e"))
print(group_words(""))


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


# ⛔ AND THE NARROWING RECORD IS PER FUNCTION. `while cur is not None` above
# left `cur`'s pre-narrowing type behind, and this function's own `cur` -- a
# different local with a different type -- was decided from it, with no
# diagnostic in between.
def scan_pairs(src: str) -> "list[str]":
    seen: "list[str]" = []
    cur = None
    for ch in src:
        if cur is None:
            cur = ch
        else:
            seen.append(cur + ch)
            cur = None
    if cur is not None:
        seen.append(cur)
    return seen


print(scan_pairs("abcde"))
