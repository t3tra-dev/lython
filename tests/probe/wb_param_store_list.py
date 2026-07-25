# probe: callee stores into a borrowed receiver's list[int] field; caller reads it back
# axes: acquire=param width=list op=rebind flow=straight observe=writeback
# CLASSIFICATION: 2 silent 誤実行
#   cpython='2\n' lyc='0\n'
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


def rebind(b: Box) -> None:
    fresh: list[int] = [1, 2]
    b.f = fresh


o = mk()
rebind(o)
print(len(o.f))
