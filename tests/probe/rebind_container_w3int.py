# probe: field rebind -- receiver read out of a list; field type int
# axes: acquire=container width=w3int op=rebind flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 123456

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v


def mk() -> Box:
    v: int = 0
    return Box(v)


boxes: list[Box] = [mk()]
o = boxes[0]
fresh: int = 123456
o.f = fresh
print(o.f)
