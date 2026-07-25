# probe: field rebind -- receiver from a method return; field type float
# axes: acquire=method width=w2float op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


class Factory:
    def make(self) -> Box:
        v: float = 0.0
        return Box(v)


f = Factory()
o = f.make()
fresh: float = 1.5
o.f = fresh
print(o.f)
