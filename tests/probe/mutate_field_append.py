# probe: in-place field mutation -- list field append (grow, may reallocate); receiver from field
# axes: acquire=field width=w3list/w1dict op=append flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   owned resource from @LyLong_Repr result 0 is released or transferred more than once on one CFG path
# CPython 3.14 expects: 2 10 20

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


class Holder:
    def __init__(self, b: Box) -> None:
        self.inner: Box = b


h = Holder(mk())
o = h.inner
o.f.append(20)
print(len(o.f), o.f[0], o.f[1])
