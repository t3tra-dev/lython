# probe: leak -- dict field rebind on a call-obtained receiver (one handle) (40000 iterations)
# axes: op=leak-loop iterations=40000
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 160000
# RSS: -54 バイト/回 → リークなし (計測ノイズ ±130 B/回 の範囲)

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {}
    return Box(v)


def once() -> int:
    o = mk()
    fresh: dict[str, int] = {"a": 1, "b": 2, "c": 3, "d": 4}
    o.f = fresh
    return len(o.f)


total = 0
for _ in range(40000):
    total += once()
print(total)
