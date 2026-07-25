# probe: REPORTED loud: an owned local live across an early return
# axes: op=local flow=earlyreturn
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 2 5

def run(flag: bool) -> int:
    xs: list[int] = [1, 2]
    if flag:
        return len(xs)
    ys: list[int] = [1, 2, 3]
    return len(xs) + len(ys)


print(run(True), run(False))
