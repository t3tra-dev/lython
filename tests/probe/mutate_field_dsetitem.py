# probe: in-place field mutation -- dict field setitem (insert, may rehash); receiver from field
# axes: acquire=field width=w3list/w1dict op=dsetitem flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 2 2

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {"a": 1}
    return Box(v)


class Holder:
    def __init__(self, b: Box) -> None:
        self.inner: Box = b


h = Holder(mk())
o = h.inner
o.f["b"] = 2
print(len(o.f), o.f["b"])
