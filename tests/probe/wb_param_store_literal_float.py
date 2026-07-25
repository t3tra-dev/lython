# probe: same store-through-parameter but the stored value is an immediate literal
# axes: acquire=param width=float op=rebind(literal) flow=straight observe=writeback
# CLASSIFICATION: 2 silent 誤実行
#   cpython='1.5\n' lyc='0.0\n'
# CPython 3.14 expects: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


def mk() -> Box:
    return Box(0.0)


def rebind(b: Box) -> None:
    b.f = 1.5


o = mk()
rebind(o)
print(o.f)
