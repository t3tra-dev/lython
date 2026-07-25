# probe: field rebind -- receiver read out of an except-bound exception's field; field type float
# axes: acquire=except width=w2float op=rebind flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


class Err(Exception):
    def __init__(self, b: Box) -> None:
        super().__init__("boom")
        self.b: Box = b


def boom() -> None:
    v: float = 0.0
    raise Err(Box(v))


try:
    boom()
except Err as e:
    o = e.b
    fresh: float = 1.5
    o.f = fresh
    print(o.f)
