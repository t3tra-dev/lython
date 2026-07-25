# probe: leak -- in-place append through self inside a method (the json append shape) (100 iterations)
# axes: op=leak-loop iterations=100
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 200
# RSS: 3 バイト/回 → リークなし (計測ノイズ ±130 B/回 の範囲)

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v

    def add(self, n: int) -> None:
        self.f.append(n)


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


def once() -> int:
    o = mk()
    o.add(2)
    return len(o.f)


total = 0
for _ in range(100):
    total += once()
print(total)
