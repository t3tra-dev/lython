# probe: leak -- list field rebind on an inline-constructed receiver (control: marker present) (100 iterations)
# axes: op=leak-loop iterations=100
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 800
# RSS: 32 バイト/回 → リークなし (計測ノイズ ±130 B/回 の範囲)

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
for _ in range(100):
    total += once()
print(total)
