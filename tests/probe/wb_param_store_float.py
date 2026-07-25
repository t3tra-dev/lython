# probe: callee stores into a borrowed receiver's float field; caller reads it back
# axes: acquire=param width=float op=rebind flow=straight observe=writeback
# CLASSIFICATION: 2 silent 誤実行
#   cpython='1.5\n' lyc='0.0\n'
# CPython 3.14 expects: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


def mk() -> Box:
    v: float = 0.0
    return Box(v)


def rebind(b: Box) -> None:
    fresh: float = 1.5
    b.f = fresh


o = mk()
rebind(o)
print(o.f)
