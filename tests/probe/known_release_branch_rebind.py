# probe: REPORTED loud: an owned local rebound inside one branch
# axes: op=local-rebind flow=ifone
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 3 1

def run(flag: bool) -> int:
    xs: list[int] = [1]
    if flag:
        xs = [1, 2, 3]
    return len(xs)


print(run(True), run(False))
