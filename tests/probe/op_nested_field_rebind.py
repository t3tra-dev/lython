# probe: rebinding a field two levels down (o.inner.f)
# axes: acquire=call width=w3list op=nested-rebind flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


class Outer:
    def __init__(self, b: Box) -> None:
        self.inner: Box = b


def mko() -> Outer:
    return Outer(mk())


o = mko()
fresh: list[int] = [1, 2, 3]
o.inner.f = fresh
print(len(o.inner.f))
