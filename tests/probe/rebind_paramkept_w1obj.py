# probe: field rebind -- receiver is a borrowed parameter, observed by the caller after return; field type Other
# axes: acquire=paramkept width=w1obj op=rebind flow=straight
# CLASSIFICATION: 2 silent 誤実行
#   cpython='7\n' lyc='0\n'
# CPython 3.14 expects: 7

class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Wide:
    def __init__(self, a: int, b: float, c: str) -> None:
        self.a: int = a
        self.b: float = b
        self.c: str = c


class Box:
    def __init__(self, v: Other) -> None:
        self.f: Other = v


def mk() -> Box:
    v: Other = Other(0)
    return Box(v)


def rebind(o: Box) -> None:
    fresh: Other = Other(7)
    o.f = fresh


held = mk()
rebind(held)
o = held
print(o.f.n)
