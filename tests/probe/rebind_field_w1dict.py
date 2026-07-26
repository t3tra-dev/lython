# probe: field rebind -- receiver read out of another object's field; field type dict[str, int]
# axes: acquire=field width=w1dict op=rebind flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {}
    return Box(v)


class Holder:
    def __init__(self, b: Box) -> None:
        self.inner: Box = b


h = Holder(mk())
o = h.inner
fresh: dict[str, int] = {"a": 1, "b": 2}
o.f = fresh
print(len(o.f), o.f["a"])
