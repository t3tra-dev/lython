# probe: in-place field mutation -- dict field setitem (insert, may rehash); receiver from call
# axes: acquire=call width=w3list/w1dict op=dsetitem flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 2 2

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {"a": 1}
    return Box(v)


o = mk()
o.f["b"] = 2
print(len(o.f), o.f["b"])
