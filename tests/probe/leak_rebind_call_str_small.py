# probe: leak -- str field rebind on a call-obtained receiver (one handle) (100 iterations)
# axes: op=leak-loop iterations=100
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 3200

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


def once() -> int:
    o = mk()
    fresh: str = "0123456789abcdef0123456789abcdef"
    o.f = fresh
    return len(o.f)


total = 0
for _ in range(100):
    total += once()
print(total)
