# probe: leak -- control: an object created and dropped each iteration, no field store (100 iterations)
# axes: op=leak-loop iterations=100
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 800
# RSS: -45 バイト/回 → リークなし (計測ノイズ ±130 B/回 の範囲)

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
for _ in range(100):
    total += once()
print(total)
