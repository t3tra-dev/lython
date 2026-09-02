# What: `[]`, `{}`, `set()` and their kin carry no element type of their own,
# so the value that fills them has to decide it -- whether that value arrives
# through an append, a later rebinding, the other arm of a choice, or a call
# site. Only printing the containers shows the elements survived.
def rebound(flag: bool) -> "list[int]":
    out = []
    if flag:
        out = [1, 2]
    return out


print(rebound(True), rebound(False))


def a_set(flag: bool) -> "set[int]":
    return set() if flag else {1, 2}


print(sorted(a_set(True)), sorted(a_set(False)))


def a_dict(flag: bool) -> "dict[str, int]":
    d = {}
    if flag:
        d = {"a": 1}
    return d


print(sorted(a_dict(True).items()), sorted(a_dict(False).items()))


def defaulted(xs=[]) -> int:
    return len(xs)


print(defaulted(), defaulted([1, 2, 3]))


def either(flag: bool) -> "list[str]":
    return [] if flag else ["x"]


print(either(True), either(False))


def grown() -> "list[int]":
    out = []
    for i in range(3):
        out.append(i)
    return out


print(grown())
