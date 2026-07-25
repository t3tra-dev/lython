# probe: the old field value is still named by a local when the field is rebound
# axes: acquire=call width=w3list op=rebind-with-live-old flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 1 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


o = mk()
old = o.f
fresh: list[int] = [1, 2, 3]
o.f = fresh
print(len(old), len(o.f))
