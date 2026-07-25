# probe: field rebind -- receiver from an inline constructor in the same frame; field type float
# axes: acquire=inline width=w2float op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


v: float = 0.0
o = Box(v)
fresh: float = 1.5
o.f = fresh
print(o.f)
