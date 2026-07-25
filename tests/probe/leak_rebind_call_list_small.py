# probe: leak -- list field rebind on a call-obtained receiver (100 iterations)
# axes: op=leak-loop iterations=100
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 800
# RSS: 8438 バイト/回 → リーク

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


def once() -> int:
    o = mk()
    fresh: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    o.f = fresh
    return len(o.f)


total = 0
for _ in range(100):
    total += once()
print(total)
