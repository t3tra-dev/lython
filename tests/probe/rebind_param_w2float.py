# probe: field rebind -- receiver is a borrowed parameter, observed inside the callee; field type float
# axes: acquire=param width=w2float op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


def mk() -> Box:
    v: float = 0.0
    return Box(v)


def rebind(o: Box) -> None:
    fresh: float = 1.5
    o.f = fresh
    print(o.f)


rebind(mk())
