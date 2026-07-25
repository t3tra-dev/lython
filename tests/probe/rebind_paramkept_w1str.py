# probe: field rebind -- receiver is a borrowed parameter, observed by the caller after return; field type str
# axes: acquire=paramkept width=w1str op=rebind flow=straight
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


held = mk()
rebind(held)
o = held
print(len(o.f), o.f)
