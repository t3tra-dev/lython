# probe: field rebind -- receiver read out of another object's field; field type Wide
# axes: acquire=field width=wNcls op=rebind flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 1 2.5 z

class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Wide:
    def __init__(self, a: int, b: float, c: str) -> None:
        self.a: int = a
        self.b: float = b
        self.c: str = c


class Box:
    def __init__(self, v: Wide) -> None:
        self.f: Wide = v


def mk() -> Box:
    v: Wide = Wide(0, 0.0, "")
    return Box(v)


class Holder:
    def __init__(self, b: Box) -> None:
        self.inner: Box = b


h = Holder(mk())
o = h.inner
fresh: Wide = Wide(1, 2.5, "z")
o.f = fresh
print(o.f.a, o.f.b, o.f.c)
