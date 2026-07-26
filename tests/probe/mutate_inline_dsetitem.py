# probe: in-place field mutation -- dict field setitem (insert, may rehash); receiver from inline
# axes: acquire=inline width=w3list/w1dict op=dsetitem flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 2 2

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {"a": 1}
    return Box(v)


v0: dict[str, int] = {"a": 1}
o = Box(v0)
o.f["b"] = 2
print(len(o.f), o.f["b"])
