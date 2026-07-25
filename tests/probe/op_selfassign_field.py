# probe: a field is rebound to its own current value
# axes: acquire=call width=w3list op=self-assign flow=straight
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 1

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


o = mk()
o.f = o.f
print(len(o.f))
