# probe: a function rebinds a field of an object it created and returns that object
# axes: acquire=inline width=w3list op=return-from-function flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


def build() -> Box:
    b = mk()
    fresh: list[int] = [1, 2, 3]
    b.f = fresh
    return b


o = build()
print(len(o.f))
