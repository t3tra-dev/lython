# probe: a function rebinds a field of an object it RECEIVED from a call and returns it
# axes: acquire=call width=w3list op=return-from-function flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 1

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


def wrap() -> Box:
    b = build()
    more: list[int] = [9]
    b.f = more
    return b


o = wrap()
print(len(o.f))
