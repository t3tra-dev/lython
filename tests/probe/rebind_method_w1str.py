# probe: field rebind -- receiver from a method return; field type str
# axes: acquire=method width=w1str op=rebind flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 4 abcd

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


class Factory:
    def make(self) -> Box:
        v: str = ""
        return Box(v)


f = Factory()
o = f.make()
fresh: str = "abcd"
o.f = fresh
print(len(o.f), o.f)
