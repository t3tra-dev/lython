# probe: leak -- in-place append to a list field of a call-obtained receiver (100 iterations)
# axes: op=leak-loop iterations=100
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 200

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


def once() -> int:
    o = mk()
    o.f.append(2)
    return len(o.f)


total = 0
for _ in range(100):
    total += once()
print(total)
