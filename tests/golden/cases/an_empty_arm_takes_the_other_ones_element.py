# What: `[]` has no element type of its own, so a conditional expression whose
# other arm has one must not join the two into a container of neither. Only
# running it shows the value each arm produced and that the elements survived.
def coalesce(xs: "list[int] | None" = None) -> "list[int]":
    return [] if xs is None else xs


print(coalesce(), coalesce([1, 2]))


def words(ws: "list[str] | None") -> "list[str]":
    got = [] if ws is None else ws
    return got


print(words(None), words(["a"]))


def table(d: "dict[str, int] | None") -> "dict[str, int]":
    return {} if d is None else d


print(sorted(table(None).items()), sorted(table({"a": 1}).items()))


def either(flag: bool) -> "list[int]":
    return [1, 2] if flag else []


print(either(True), either(False))


def both_empty(flag: bool) -> "list[int]":
    out: "list[int]" = [] if flag else []
    out.append(3)
    return out


print(both_empty(True), both_empty(False))
