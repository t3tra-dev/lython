# probe: field rebind -- receiver read out of a list; field type float
# axes: acquire=container width=w2float op=rebind flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


def mk() -> Box:
    v: float = 0.0
    return Box(v)


boxes: list[Box] = [mk()]
o = boxes[0]
fresh: float = 1.5
o.f = fresh
print(o.f)
