# probe: leak -- int field rebind on a call-obtained receiver (three lanes, no heap payload for small ints) (100 iterations)
# axes: op=leak-loop iterations=100
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 123456700

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v


def mk() -> Box:
    return Box(0)


def once() -> int:
    o = mk()
    fresh: int = 1234567
    o.f = fresh
    return o.f


total = 0
for _ in range(100):
    total += once()
print(total)
