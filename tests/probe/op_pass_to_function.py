# probe: an object whose field was rebound is then passed to a function
# axes: acquire=call width=w3list op=pass-to-function flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


def take(b: Box) -> int:
    return len(b.f)


o = mk()
fresh: list[int] = [1, 2, 3]
o.f = fresh
print(take(o))
