# probe: in-place field mutation -- dict field del (in place); receiver from field
# axes: acquire=field width=w3list/w1dict op=ddel flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 0

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
del o.f["a"]
print(len(o.f))
