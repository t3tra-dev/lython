# probe: field rebind -- receiver from a method return; field type list[int]
# axes: acquire=method width=w3list op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


class Factory:
    def make(self) -> Box:
        v: list[int] = []
        return Box(v)


f = Factory()
o = f.make()
fresh: list[int] = [1, 2]
o.f = fresh
print(len(o.f), o.f[0])
