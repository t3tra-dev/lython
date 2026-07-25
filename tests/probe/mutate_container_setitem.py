# probe: in-place field mutation -- list field setitem (in place, no realloc); receiver from container
# axes: acquire=container width=w3list/w1dict op=setitem flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 1 99

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


boxes: list[Box] = [mk()]
o = boxes[0]
o.f[0] = 99
print(len(o.f), o.f[0])
