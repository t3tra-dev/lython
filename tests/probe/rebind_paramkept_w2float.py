# probe: field rebind -- receiver is a borrowed parameter, observed by the caller after return; field type float
# axes: acquire=paramkept width=w2float op=rebind flow=straight
# CLASSIFICATION: 2 silent 誤実行
#   cpython='1.5\n' lyc='0.0\n'
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


held = mk()
rebind(held)
o = held
print(o.f)
