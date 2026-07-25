# probe: field rebind -- receiver read out of another object's field; field type list[int]
# axes: acquire=field width=w3list op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


class Holder:
    def __init__(self, b: Box) -> None:
        self.inner: Box = b


h = Holder(mk())
o = h.inner
fresh: list[int] = [1, 2]
o.f = fresh
print(len(o.f), o.f[0])
