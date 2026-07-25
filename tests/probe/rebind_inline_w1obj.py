# probe: field rebind -- receiver from an inline constructor in the same frame; field type Other
# axes: acquire=inline width=w1obj op=rebind flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 7

class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Wide:
    def __init__(self, a: int, b: float, c: str) -> None:
        self.a: int = a
        self.b: float = b
        self.c: str = c


class Box:
    def __init__(self, v: Other) -> None:
        self.f: Other = v


v: Other = Other(0)
o = Box(v)
fresh: Other = Other(7)
o.f = fresh
print(o.f.n)
