# probe: callee stores into and reads back a borrowed receiver's complex field, then the caller reads it
# axes: acquire=param width=complex op=rebind flow=straight observe=writeback-both
# CLASSIFICATION: 2 silent 誤実行
#   cpython='in callee: (1+2j)\nin caller: (1+2j)\n' lyc='in callee: (1+2j)\nin caller: 0j\n'
# CPython 3.14 expects: in callee: (1+2j) / in caller: (1+2j)

class Box:
    def __init__(self, v: complex) -> None:
        self.f: complex = v


def mk() -> Box:
    v: complex = 0j
    return Box(v)


def rebind(b: Box) -> None:
    fresh: complex = 1 + 2j
    b.f = fresh
    o = b
    print("in callee:", o.f)


o = mk()
rebind(o)
print("in caller:", o.f)
