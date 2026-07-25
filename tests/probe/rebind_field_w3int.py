# probe: field rebind -- receiver read out of another object's field; field type int
# axes: acquire=field width=w3int op=rebind flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 123456

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v


def mk() -> Box:
    v: int = 0
    return Box(v)


class Holder:
    def __init__(self, b: Box) -> None:
        self.inner: Box = b


h = Holder(mk())
o = h.inner
fresh: int = 123456
o.f = fresh
print(o.f)
