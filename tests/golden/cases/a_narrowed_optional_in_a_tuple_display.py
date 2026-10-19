# A value narrowed out of `str | None` placed in a tuple display. The display
# took the union's own reference as if the value were a temporary, so the
# program was refused ("owned resource ... reaches function exit without
# release") while a list display of the same value compiled.
# Why run it: a reference moved twice or not at all shows as the wrong string
# (or a crash) only when the tuple and the name are both read afterwards.


def short(flag: bool) -> str | None:
    if flag:
        return "described" + str(flag)
    return None


def joined(flag: bool) -> str:
    doc = short(flag)
    if doc is not None:
        return "\n".join(("case", doc))
    return "case"


def second(flag: bool) -> str:
    doc = short(flag)
    if doc is not None:
        pair = ("case", doc)
        return pair[1] + "/" + doc
    return "none"


def single(flag: bool) -> str:
    doc = short(flag)
    if doc is not None:
        return " ".join((doc,))
    return "none"


print(joined(True))
print(joined(False))
print(second(True), second(False))
print(single(True), single(False))
total = 0
for i in range(2000):
    total += len(second(i % 2 == 0)) + len(joined(i % 3 == 0))
print(total)
