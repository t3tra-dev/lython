# probe: the store happens two call levels deep through borrowed receivers (float field)
# axes: acquire=param width=float op=rebind flow=straight observe=writeback
# CLASSIFICATION: 2 silent 誤実行
#   cpython='1.5\n' lyc='0.0\n'
# CPython 3.14 expects: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


def mk() -> Box:
    return Box(0.0)


def inner(b: Box) -> None:
    fresh: float = 1.5
    b.f = fresh


def outer(b: Box) -> None:
    inner(b)


o = mk()
outer(o)
print(o.f)
