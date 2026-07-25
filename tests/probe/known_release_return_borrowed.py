# probe: REPORTED loud: a helper that returns a borrowed value
# axes: op=return-borrowed flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 2

def first(xs: list[list[int]]) -> list[int]:
    return xs[0]


data: list[list[int]] = [[1, 2], [3]]
print(len(first(data)))
