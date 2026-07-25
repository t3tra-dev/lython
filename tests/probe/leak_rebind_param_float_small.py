# probe: leak -- float field rebind through a borrowed parameter (the SILENT shape, checked for leak too) (100 iterations)
# axes: op=leak-loop iterations=100
# CLASSIFICATION: 2 silent 誤実行
#   cpython='100\n' lyc='0\n'
# CPython 3.14 expects: 100
# RSS: -25 バイト/回 → リークなし (計測ノイズ ±130 B/回 の範囲)

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


def mk() -> Box:
    return Box(0.0)


def rebind(o: Box) -> None:
    fresh: float = 1.5
    o.f = fresh


def once() -> int:
    held = mk()
    rebind(held)
    if held.f > 1.0:
        return 1
    return 0


total = 0
for _ in range(100):
    total += once()
print(total)
