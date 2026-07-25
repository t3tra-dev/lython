# probe: in-place field mutation -- list field extend; receiver from field
# axes: acquire=field width=w3list/w1dict op=extend flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 3 loud 拒否 (診断)
#   runtime bundle for '!py.contract<"types.NoneType">' has 3 values, but ABI expects 0
# CPython 3.14 expects: 3 3

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
o.f.extend([2, 3])
print(len(o.f), o.f[2])
