# probe: an object obtained from a call is stored into a dict, then mutated through the dict
# axes: acquire=call width=w3list op=store-into-container flow=straight
# CLASSIFICATION: 2 silent 誤実行
#   cpython='2 2\n' lyc='1 1\n'
# CPython 3.14 expects: 2 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


o = mk()
d: dict[str, Box] = {}
d["k"] = o
fresh: list[int] = [7, 8]
d["k"].f = fresh
print(len(d["k"].f), len(o.f))
