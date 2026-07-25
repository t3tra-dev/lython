# probe: field rebind -- receiver from a method return; field type int
# axes: acquire=method width=w3int op=rebind flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 123456

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v


class Factory:
    def make(self) -> Box:
        v: int = 0
        return Box(v)


f = Factory()
o = f.make()
fresh: int = 123456
o.f = fresh
print(o.f)
