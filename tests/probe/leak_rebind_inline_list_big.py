# probe: leak -- list field rebind on an inline-constructed receiver (control: marker present) (40000 iterations)
# axes: op=leak-loop iterations=40000
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 320000
# RSS: 35 バイト/回 → リークなし (計測ノイズ ±130 B/回 の範囲)

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def once() -> int:
    v: list[int] = []
    o = Box(v)
    fresh: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    o.f = fresh
    return len(o.f)


total = 0
for _ in range(40000):
    total += once()
print(total)
