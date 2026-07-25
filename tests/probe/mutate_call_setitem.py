# probe: in-place field mutation -- list field setitem (in place, no realloc); receiver from call
# axes: acquire=call width=w3list/w1dict op=setitem flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 1 99

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


o = mk()
o.f[0] = 99
print(len(o.f), o.f[0])
