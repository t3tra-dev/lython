# probe: REPORTED loud: an owned local consumed on only one arm of an if
# axes: op=consume flow=ifone
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 2 0

def sink(xs: list[int]) -> int:
    return len(xs)


def run(flag: bool) -> int:
    xs: list[int] = [1, 2]
    if flag:
        return sink(xs)
    return 0


print(run(True), run(False))
