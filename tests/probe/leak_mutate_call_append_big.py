# probe: leak -- in-place append to a list field of a call-obtained receiver (40000 iterations)
# axes: op=leak-loop iterations=40000
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 80000
# RSS: -74 バイト/回 → リークなし (計測ノイズ ±130 B/回 の範囲)

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
for _ in range(40000):
    total += once()
print(total)
