# probe: leak -- control: an object created and dropped each iteration, no field store (40000 iterations)
# axes: op=leak-loop iterations=40000
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 320000

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    return Box(v)


def once() -> int:
    o = mk()
    return len(o.f)


total = 0
for _ in range(40000):
    total += once()
print(total)
