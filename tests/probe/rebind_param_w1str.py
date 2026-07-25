# probe: field rebind -- receiver is a borrowed parameter, observed inside the callee; field type str
# axes: acquire=param width=w1str op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 4 abcd

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


def rebind(o: Box) -> None:
    fresh: str = "abcd"
    o.f = fresh
    print(len(o.f), o.f)


rebind(mk())
