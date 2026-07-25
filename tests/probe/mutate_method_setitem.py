# probe: in-place field mutation -- list field setitem (in place, no realloc); receiver from method
# axes: acquire=method width=w3list/w1dict op=setitem flow=straight
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 1 99

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


class Factory:
    def make(self) -> Box:
        return mk()


fac = Factory()
o = fac.make()
o.f[0] = 99
print(len(o.f), o.f[0])
