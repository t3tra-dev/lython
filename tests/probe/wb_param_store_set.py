# probe: callee stores into a borrowed receiver's set[int] field; caller reads it back
# axes: acquire=param width=set op=rebind flow=straight observe=writeback
# CLASSIFICATION: 2 silent 誤実行
#   cpython='2\n' lyc='0\n'
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: set[int]) -> None:
        self.f: set[int] = v


def mk() -> Box:
    v: set[int] = set()
    return Box(v)


def rebind(b: Box) -> None:
    fresh: set[int] = {1, 2}
    b.f = fresh


o = mk()
rebind(o)
print(len(o.f))
