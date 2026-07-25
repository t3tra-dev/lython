# probe: field rebind -- receiver from a method return; field type dict[str, int]
# axes: acquire=method width=w1dict op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


class Factory:
    def make(self) -> Box:
        v: dict[str, int] = {}
        return Box(v)


f = Factory()
o = f.make()
fresh: dict[str, int] = {"a": 1, "b": 2}
o.f = fresh
print(len(o.f), o.f["a"])
