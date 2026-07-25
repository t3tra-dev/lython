# probe: field rebind -- receiver is a borrowed parameter, observed by the caller after return; field type dict[str, int]
# axes: acquire=paramkept width=w1dict op=rebind flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {}
    return Box(v)


def rebind(o: Box) -> None:
    fresh: dict[str, int] = {"a": 1, "b": 2}
    o.f = fresh


held = mk()
rebind(held)
o = held
print(len(o.f), o.f["a"])
