# probe: in-place mutation through self inside a method -- list field extend
# axes: acquire=self width=w3list/w1dict op=extend flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 3 loud 拒否 (診断)
#   runtime bundle for '!py.contract<"types.NoneType">' has 3 values, but ABI expects 0
# CPython 3.14 expects: 3 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v

    def touch(self) -> None:
        self.f.extend([2, 3])

    def show(self) -> None:
        print(len(self.f), self.f[2])


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


o = mk()
o.touch()
o.show()
