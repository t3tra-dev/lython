# probe: a field is rebound to the value read out of another object's field (shared value)
# axes: acquire=call width=w3list op=field-to-field flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 3 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


a = mk()
b = mk()
extra: list[int] = [1, 2, 3]
a.f = extra
b.f = a.f
print(len(a.f), len(b.f))
