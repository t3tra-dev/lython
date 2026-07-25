# probe: field rebind -- receiver read out of an except-bound exception's field; field type dict[str, int]
# axes: acquire=except width=w1dict op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


class Err(Exception):
    def __init__(self, b: Box) -> None:
        super().__init__("boom")
        self.b: Box = b


def boom() -> None:
    v: dict[str, int] = {}
    raise Err(Box(v))


try:
    boom()
except Err as e:
    o = e.b
    fresh: dict[str, int] = {"a": 1, "b": 2}
    o.f = fresh
    print(len(o.f), o.f["a"])
