# probe: field rebind -- receiver read out of an except-bound exception's field; field type list[int]
# axes: acquire=except width=w3list op=rebind flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


class Err(Exception):
    def __init__(self, b: Box) -> None:
        super().__init__("boom")
        self.b: Box = b


def boom() -> None:
    v: list[int] = []
    raise Err(Box(v))


try:
    boom()
except Err as e:
    o = e.b
    fresh: list[int] = [1, 2]
    o.f = fresh
    print(len(o.f), o.f[0])
