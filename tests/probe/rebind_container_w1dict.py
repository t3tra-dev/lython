# probe: field rebind -- receiver read out of a list; field type dict[str, int]
# axes: acquire=container width=w1dict op=rebind flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {}
    return Box(v)


boxes: list[Box] = [mk()]
o = boxes[0]
fresh: dict[str, int] = {"a": 1, "b": 2}
o.f = fresh
print(len(o.f), o.f["a"])
