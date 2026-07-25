# probe: field rebind -- receiver is a borrowed parameter, observed inside the callee; field type int
# axes: acquire=param width=w3int op=rebind flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 123456

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v


def mk() -> Box:
    v: int = 0
    return Box(v)


def rebind(o: Box) -> None:
    fresh: int = 123456
    o.f = fresh
    print(o.f)


rebind(mk())
