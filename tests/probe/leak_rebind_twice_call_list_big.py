# probe: leak -- two successive list field rebinds on a call-obtained receiver (40000 iterations)
# axes: op=leak-loop iterations=40000
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 320000
# RSS: 17176 バイト/回 → リーク

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


def once() -> int:
    o = mk()
    a: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    o.f = a
    b: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    o.f = b
    return len(o.f)


total = 0
for _ in range(40000):
    total += once()
print(total)
