# probe: field rebind -- receiver from a function return; field type float
# axes: acquire=call width=w2float op=rebind flow=straight
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


def mk() -> Box:
    v: float = 0.0
    return Box(v)


o = mk()
fresh: float = 1.5
o.f = fresh
print(o.f)
