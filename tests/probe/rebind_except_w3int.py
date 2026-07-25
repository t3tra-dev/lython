# probe: field rebind -- receiver read out of an except-bound exception's field; field type int
# axes: acquire=except width=w3int op=rebind flow=straight
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 123456

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v


class Err(Exception):
    def __init__(self, b: Box) -> None:
        super().__init__("boom")
        self.b: Box = b


def boom() -> None:
    v: int = 0
    raise Err(Box(v))


try:
    boom()
except Err as e:
    o = e.b
    fresh: int = 123456
    o.f = fresh
    print(o.f)
