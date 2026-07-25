# probe: field rebind -- receiver read out of an except-bound exception's field; field type Other
# axes: acquire=except width=w1obj op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
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


class Err(Exception):
    def __init__(self, b: Box) -> None:
        super().__init__("boom")
        self.b: Box = b


def boom() -> None:
    v: Other = Other(0)
    raise Err(Box(v))


try:
    boom()
except Err as e:
    o = e.b
    fresh: Other = Other(7)
    o.f = fresh
    print(o.f.n)
