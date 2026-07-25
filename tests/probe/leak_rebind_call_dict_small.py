# probe: leak -- dict field rebind on a call-obtained receiver (one handle) (100 iterations)
# axes: op=leak-loop iterations=100
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 400
# RSS: 17 バイト/回 → リークなし (計測ノイズ ±80 B/回 の範囲)

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
for _ in range(100):
    total += once()
print(total)
