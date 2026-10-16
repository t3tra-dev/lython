# Why execution: the returned parameter must hold a reference of its own --
# the caller reads its argument again after the call, and the leak gate counts
# the retain -- which only running shows.
#
# A parameter returned through a `finally` as an Optional merges with the
# None placeholder the not-returning path carries in the completion payload.
# The merge was neither the parameter nor owned, so no retain went in and the
# program was refused: "borrowed entry argument 0 of @pick is returned as
# owned without a dominating retain".
def pick(flag: int) -> "int | None":
    try:
        return flag
    finally:
        print("checked")


def name_or_none(s: str, keep: bool) -> "str | None":
    try:
        if keep:
            return s
        return None
    finally:
        print("named", keep)


def guarded(s: str) -> "str | None":
    try:
        if len(s) > 3:
            raise ValueError(s)
        return s
    except ValueError:
        return None
    finally:
        print("guarded", s)


def through_loop(xs: list[int]) -> "list[int] | None":
    try:
        for i in range(len(xs)):
            xs[i] += 1
        return xs
    finally:
        print("looped")


big = 10 ** 30
print(pick(1), pick(big))
for _ in range(3):
    print(name_or_none("w" + str(_), True), name_or_none("x", False))
print(guarded("ab"), guarded("abcdef"))
print(through_loop([1, 2]))
word = "kept" + str(len(big.__str__()))
print(name_or_none(word, True), word)
