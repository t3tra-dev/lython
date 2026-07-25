# probe: field rebind -- receiver read out of an except-bound exception's field; field type str
# axes: acquire=except width=w1str op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 4 abcd

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


class Err(Exception):
    def __init__(self, b: Box) -> None:
        super().__init__("boom")
        self.b: Box = b


def boom() -> None:
    v: str = ""
    raise Err(Box(v))


try:
    boom()
except Err as e:
    o = e.b
    fresh: str = "abcd"
    o.f = fresh
    print(len(o.f), o.f)
