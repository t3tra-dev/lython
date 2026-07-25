# probe: callee stores into a borrowed receiver's bool field; caller reads it back
# axes: acquire=param width=bool op=rebind flow=straight observe=writeback
# CLASSIFICATION: 2 silent 誤実行
#   cpython='True\n' lyc='False\n'
# CPython 3.14 expects: True

class Box:
    def __init__(self, v: bool) -> None:
        self.f: bool = v


def mk() -> Box:
    v: bool = False
    return Box(v)


def rebind(b: Box) -> None:
    fresh: bool = True
    b.f = fresh


o = mk()
rebind(o)
print(o.f)
