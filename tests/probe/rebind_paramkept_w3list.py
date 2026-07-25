# probe: field rebind -- receiver is a borrowed parameter, observed by the caller after return; field type list[int]
# axes: acquire=paramkept width=w3list op=rebind flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   IndexError: sequence index out of range
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


def rebind(o: Box) -> None:
    fresh: list[int] = [1, 2]
    o.f = fresh


held = mk()
rebind(held)
o = held
print(len(o.f), o.f[0])
