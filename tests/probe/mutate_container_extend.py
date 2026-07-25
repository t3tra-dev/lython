# probe: in-place field mutation -- list field extend; receiver from container
# axes: acquire=container width=w3list/w1dict op=extend flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 3 loud 拒否 (診断)
#   runtime bundle for '!py.contract<"types.NoneType">' has 3 values, but ABI expects 0
# CPython 3.14 expects: 3 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


boxes: list[Box] = [mk()]
o = boxes[0]
o.f.extend([2, 3])
print(len(o.f), o.f[2])
