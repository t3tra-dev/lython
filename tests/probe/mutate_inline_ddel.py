# probe: in-place field mutation -- dict field del (in place); receiver from inline
# axes: acquire=inline width=w3list/w1dict op=ddel flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 0

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {"a": 1}
    return Box(v)


v0: dict[str, int] = {"a": 1}
o = Box(v0)
del o.f["a"]
print(len(o.f))
