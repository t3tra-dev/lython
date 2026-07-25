# probe: augmented assignment to a borrowed receiver's float field
# axes: acquire=param width=float op=augassign flow=straight observe=writeback
# CLASSIFICATION: 2 silent 誤実行
#   cpython='1.5\n' lyc='1.0\n'
# CPython 3.14 expects: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


def mk() -> Box:
    return Box(1.0)


def bump(b: Box) -> None:
    b.f += 0.5


o = mk()
bump(o)
print(o.f)
