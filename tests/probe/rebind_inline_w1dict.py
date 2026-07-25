# probe: field rebind -- receiver from an inline constructor in the same frame; field type dict[str, int]
# axes: acquire=inline width=w1dict op=rebind flow=straight
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


v: dict[str, int] = {}
o = Box(v)
fresh: dict[str, int] = {"a": 1, "b": 2}
o.f = fresh
print(len(o.f), o.f["a"])
