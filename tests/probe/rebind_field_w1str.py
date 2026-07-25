# probe: field rebind -- receiver read out of another object's field; field type str
# axes: acquire=field width=w1str op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 4 abcd

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


class Holder:
    def __init__(self, b: Box) -> None:
        self.inner: Box = b


h = Holder(mk())
o = h.inner
fresh: str = "abcd"
o.f = fresh
print(len(o.f), o.f)
