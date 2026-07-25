# probe: read an object out of a list into a local, then mutate the local's field
# axes: acquire=container width=w3list op=read-then-mutate flow=straight
# CLASSIFICATION: 2 silent 誤実行
#   cpython='3 1\n' lyc='1 1\n'
# CPython 3.14 expects: 3 1

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


xs: list[Box] = [mk(), mk()]
o = xs[1]
fresh: list[int] = [4, 5, 6]
o.f = fresh
print(len(xs[1].f), len(xs[0].f))
