# probe: field rebind -- receiver read out of a list; field type str
# axes: acquire=container width=w1str op=rebind flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 4 abcd

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


boxes: list[Box] = [mk()]
o = boxes[0]
fresh: str = "abcd"
o.f = fresh
print(len(o.f), o.f)
