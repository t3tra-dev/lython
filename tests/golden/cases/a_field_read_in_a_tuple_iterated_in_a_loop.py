# A field read placed in a tuple display that `join` or `for` iterates, in a
# loop. The read holds a reference of its own and the iteration reads each
# element back through an `scf.if` arm per index; the release placement
# stopped at that arm's yield and placed nothing, so the reference reached
# the exit still owned. Outside a loop a fallback before the return hid it.
# Why run it: the reference must outlive every read of the element and be
# dropped once per trip -- the joined strings and the lengths say both.


class Holder:
    def __init__(self, description: str) -> None:
        self.description = description


def joined(holders: list[Holder]) -> str:
    out = ""
    for h in holders:
        out += "-".join((h.description, "x")) + ";"
    return out


def lengths(holders: list[Holder]) -> int:
    n = 0
    for h in holders:
        for s in (h.description, h.description + "y"):
            n += len(s)
    return n


holders = [Holder("a"), Holder("bb"), Holder("ccc")]
print(joined(holders))
print(lengths(holders))
total = 0
for i in range(2000):
    total += lengths(holders) + len(joined(holders))
print(total)
