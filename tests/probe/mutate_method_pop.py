# probe: in-place field mutation -- list field pop (shrink); receiver from method
# axes: acquire=method width=w3list/w1dict op=pop flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 0

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
o.f.pop()
print(len(o.f))
