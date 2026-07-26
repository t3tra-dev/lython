# probe: leak -- str field rebind on a call-obtained receiver (one handle) (40000 iterations)
# axes: op=leak-loop iterations=40000
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 1280000

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
for _ in range(40000):
    total += once()
print(total)
