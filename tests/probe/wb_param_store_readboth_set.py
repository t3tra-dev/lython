# probe: callee stores into and reads back a borrowed receiver's set[int] field, then the caller reads it
# axes: acquire=param width=set op=rebind flow=straight observe=writeback-both
# CLASSIFICATION: 2 silent 誤実行
#   cpython='in callee: 2\nin caller: 2\n' lyc='in callee: 2\nin caller: 0\n'
# CPython 3.14 expects: in callee: 2 / in caller: 2

class Box:
    def __init__(self, v: set[int]) -> None:
        self.f: set[int] = v


def mk() -> Box:
    v: set[int] = set()
    return Box(v)


def rebind(b: Box) -> None:
    fresh: set[int] = {1, 2}
    b.f = fresh
    o = b
    print("in callee:", len(o.f))


o = mk()
rebind(o)
print("in caller:", len(o.f))
