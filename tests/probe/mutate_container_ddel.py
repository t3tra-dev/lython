# probe: in-place field mutation -- dict field del (in place); receiver from container
# axes: acquire=container width=w3list/w1dict op=ddel flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 0

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {"a": 1}
    return Box(v)


boxes: list[Box] = [mk()]
o = boxes[0]
del o.f["a"]
print(len(o.f))
