# probe: in-place field mutation -- list field pop (shrink); receiver from inline
# axes: acquire=inline width=w3list/w1dict op=pop flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 0

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


v0: list[int] = [10]
o = Box(v0)
o.f.pop()
print(len(o.f))
