# Why execution: the values are what the merged type has to keep usable. This
# is the canonical Optional idiom and it did not compile:
#
#     def f(n: int | None = None) -> int:
#         if n is None:
#             n = 0
#         return n + 1     # union<int, None> does not provide '__add__'
#
# Two things were missing. The fall-through edge of an else-less `if` carried
# the UNNARROWED outer type, so the None stayed in the join even though that
# edge is exactly where the condition is false -- `if n is None: return 0`
# worked only because that edge never reaches the join. And an EMPTY container
# literal assigned there took `list[object]`: the flow type of the name inside
# the branch is None (right for a read, no constraint on a write), so the
# expectation has to be the type the narrowing replaced.
def scalar(n: int | None = None) -> int:
    if n is None:
        n = 0
    return n + 1


def text(s: str | None = None) -> str:
    if s is None:
        s = "d"
    return s.upper()


def items(xs: list[int] | None = None) -> list[int]:
    if xs is None:
        xs = []
    xs.append(1)
    return xs


def mapping(d: dict[str, int] | None = None) -> int:
    if d is None:
        d = {}
    d["a"] = 1
    return len(d)


def both_arms(s: str | None) -> str:
    if s is not None:
        s = s.upper()
    else:
        s = "none"
    return s


# The conditional expression is the same fact with no statement to hang it
# on: each arm sees what its side of the test proves.
def picked(n: int | None) -> int:
    return (0 if n is None else n) + 1


def kept(n: int | None) -> int:
    m = n if n is not None else 0
    return m + 1


def main() -> None:
    print(scalar(), scalar(9))
    print(text(), text("x"))
    print(items(), items([9]))
    print(mapping(), mapping({"z": 0}))
    print(both_arms(None), both_arms("x"))
    # the mutable default is not shared between calls
    print(items(), items())
    print(picked(None), picked(9), kept(None), kept(9))


main()
